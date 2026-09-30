"""Pre-registered, paired development experiment for GUFU-inspired map updates.

U labels only exist in the evaluator (split auditing and explicitly labelled
oracle diagnostics). Official validation is opened after selection.json exists.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from data.uji import load_floors
from models.cross_floor import prior_map
from models.radio_map import cells
from models.unlabeled_map import prepare_update, apply_update
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import episodes, mde
from scripts.evaluate_transfer_map import transfer_prior, summary

STRENGTHS = (0.0, 0.1, 1.0, 10.0)
FIELD, MATCH = (40.0, 0.3), (12.0, -85.0)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_unlabeled(floor, support, remaining, seed):
    """Partition by position for benchmarking; returns indices, never model labels."""
    rng = np.random.default_rng(seed)
    xy = floor['xy']
    keys, inv = np.unique(xy[remaining], axis=0, return_inverse=True)
    inv = inv.ravel()
    chosen = rng.permutation(len(keys))[:len(keys)//2]
    u = remaining[np.isin(inv, chosen)]
    q = remaining[~np.isin(inv, chosen)]
    u = rng.choice(u, min(600, len(u)), replace=False)
    q = rng.choice(q, min(300, len(q)), replace=False)
    sets = [set(map(tuple, xy[idx])) for idx in (support, u, q)]
    assert all(not sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    # UJI duplicate observations at the same position cannot cross these sets.
    return u, q


def diagnostic_variants(prepared, unlabeled_xy, seed):
    shuffled = dict(prepared)
    shuffled['h'] = prepared['h'][np.random.default_rng(seed).permutation(len(unlabeled_xy))]
    oracle = dict(prepared)
    p = prepared['positions'][:len(prepared['f0'])]
    idx = ((unlabeled_xy[:, None] - p[None])**2).sum(-1).argmin(1)
    oracle['h'] = np.eye(len(p))[idx]
    return {'shuffled_eta1': shuffled, 'oracle_extra_labels_eta1': oracle}


def floor_episodes(floor, base, repeats, seed, strengths, validation=None, controls=False):
    output = []
    for ep, (s, rest) in enumerate(episodes(floor['xy'], 10, repeats, 3, np.random.default_rng(seed))):
        # Independent stream: never changes the original TPM support sequence.
        u, q = split_unlabeled(floor, s, rest, seed * 10000 + ep)
        prepared = prepare_update(base, floor['rssi'][s], floor['xy'][s], floor['rssi'][u], FIELD, MATCH)
        p = base[1]
        inferred = prepared['h'] @ p
        spatial_var = np.maximum(prepared['h'] @ (p*p).sum(1) - (inferred*inferred).sum(1), 0)
        row = {'episode': ep, 'support_row_ids': floor['row_ids'][s].tolist(),
               'unlabeled_row_ids': floor['row_ids'][u].tolist(), 'query_row_ids': floor['row_ids'][q].tolist(),
               'association_mde': mde(inferred, floor['xy'][u]),
               'association_spread_m': float(np.sqrt(spatial_var).mean()), 'methods': {}}
        variants = {f'eta{g:g}': (prepared, g) for g in strengths}
        if controls:
            variants.update({k: (v, 1.0) for k, v in diagnostic_variants(prepared, floor['xy'][u], ep+21).items()})
        for name, (pr, strength) in variants.items():
            cand, diagnostics = apply_update(pr, strength)
            rec = {'query_mde': mde(tobit_match(floor['rssi'][q], *cand, *MATCH), floor['xy'][q]),
                   'diagnostics': diagnostics}
            if validation is not None:
                rec['validation_mde'] = mde(tobit_match(validation['rssi'], *cand, *MATCH), validation['xy'])
            row['methods'][name] = rec
        output.append(row)
    return output


def means(rows, metric):
    return {name: float(np.mean([row['methods'][name][metric] for row in rows]))
            for name in rows[0]['methods']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/signal_calibrated_map.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--development-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(args.config).read_text())
    source_files = ['models/unlabeled_map.py', 'models/cross_floor.py', 'models/radio_map.py',
                    'scripts/evaluate_unlabeled_map.py', 'scripts/evaluate_scm_tobit.py',
                    'scripts/evaluate_signal_calibrated_map.py', 'scripts/evaluate_transfer_map.py', 'data/uji.py']
    meta = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_sha256': {s: sha(s) for s in source_files},
            'train_sha256': sha(cfg['train_path']), 'episodes': args.episodes,
            'field': FIELD, 'match': MATCH, 'strengths': STRENGTHS,
            'query_protocol': 'position-disjoint support / U / Q; validation never adapted on',
            'numpy_version': np.__version__}
    save(args.output/'provenance.json', meta)
    train = load_floors(cfg['train_path'])
    floors = sorted({f for pair in PAIRS for f in pair})
    c = {f: cells(train[f]) for f in floors}
    bases, priors, development = {}, {}, {}
    for lo, up in PAIRS:
        print('fit prior excluding', up, flush=True)
        tf = transfer_prior(c, floors, up)
        bases[up] = prior_map(c[lo], tf)
        priors[up] = summary(tf)
        np.savez_compressed(args.output/f'base_{up}.npz', fingerprints=bases[up][0], positions=bases[up][1])
        rows = floor_episodes(train[up], bases[up], args.episodes, 1, STRENGTHS, controls=True)
        save(args.output/f'development_{up}.json', rows)
        development[up] = means(rows, 'query_mde')
        print('development', up, development[up], flush=True)
    equal = {k: float(np.mean([v[k] for v in development.values()])) for k in development[floors[1]]}
    selected = min(STRENGTHS, key=lambda g: equal[f'eta{g:g}'])
    selection = {'selected_strength': selected, 'development_per_floor': development,
                 'development_equal': equal, 'priors': priors,
                 'note': 'development selection, not nested unseen-floor evaluation; controls excluded from selection'}
    save(args.output/'selection.json', selection)
    print('FROZEN', selected, equal, flush=True)
    if args.development_only:
        return
    # Only after model selection has been persisted do we read validation.
    validation = load_floors(cfg['validation_path'])
    final, internal = {}, {}
    for _, up in PAIRS:
        rows = floor_episodes(train[up], bases[up], args.episodes, 1001,
                              sorted(set([0., selected])), validation[up], controls=True)
        save(args.output/f'validation_{up}.json', rows)
        final[up], internal[up] = means(rows, 'validation_mde'), means(rows, 'query_mde')
        print('validation', up, final[up], flush=True)
    result = {'selection_sha256': sha(args.output/'selection.json'),
              'validation_sha256': sha(cfg['validation_path']), 'selected_strength': selected,
              'validation_per_floor': final, 'query_per_floor': internal,
              'validation_equal': {k: float(np.mean([r[k] for r in final.values()])) for k in final[PAIRS[0][1]]},
              'query_equal': {k: float(np.mean([r[k] for r in internal.values()])) for k in internal[PAIRS[0][1]]}}
    save(args.output/'results.json', result)
    print('DONE', result['validation_equal'], result['query_equal'], flush=True)


if __name__ == '__main__':
    main()
