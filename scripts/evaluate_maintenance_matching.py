"""Matched maintenance mechanisms on the already-observed temporal manifest.

The four baseline cells keep GP and matcher parameters identical. Group-prior
models add log(pi)-log(K) and log(1-pi)-log(Nold) before a global softmax.
This is a conditional energy model with explicit candidate-group priors, not
a physical RSS generative likelihood or a complete spatial-density correction.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp

from data.uji import load_floors
from scripts.evaluate_signal_calibrated_map import dbm
from scripts.evaluate_scm_tobit import tobit_match
from scripts.evaluate_temporal_mogp import RAW, COUNTS, old_context, maps

PRIOR = Path('results/temporal_mogp')
PIS = (.01, .05, .1, .25, .5)
METHODS = ('old', 'append', 'field_only', 'field_append', 'group_append', 'group_field_append')


def energy(qraw, prototypes, sigma, theta):
    """Literal frozen Tobit-inspired score: both observed and missing AP sums."""
    det = (qraw != 100).astype(float)
    x = dbm(qraw) * det
    d2 = (x*x).sum(1)[:, None] + det @ (prototypes*prototypes).T - 2*x @ prototypes.T
    return -.5 / sigma**2 * d2 + (1-det) @ log_ndtr((theta-prototypes)/sigma).T


def posterior(qraw, prototypes, nold, matcher, pi=None):
    score = energy(qraw, prototypes, *matcher)
    nnew = len(prototypes) - nold
    if pi is not None and nnew:
        # Endpoints are mathematical controls, never source-grid choices.
        score[:, :nold] += np.log1p(-pi) - np.log(nold) if pi < 1 else -np.inf
        score[:, nold:] += np.log(pi) - np.log(nnew) if pi > 0 else -np.inf
    return np.exp(score - logsumexp(score, axis=1, keepdims=True))


def four_maps(context, sr, sx, params):
    old, nodes = context[:2]
    if len(sr) == 0:
        return {m: (old, nodes) for m in METHODS}
    anchor = dbm(sr)
    # Reuse exactly the prior frozen GP update, clipping and exact-hit lookup.
    full, full_xy = maps(context, sr, sx, params, 'scm')
    field = full[:len(old)]
    return {'old': (old, nodes), 'append': (np.vstack((old, anchor)), np.vstack((nodes, sx))),
            'field_only': (field, nodes), 'field_append': (full, full_xy),
            'group_append': (np.vstack((old, anchor)), np.vstack((nodes, sx))),
            'group_field_append': (full, full_xy)}


def evaluate(qr, qx, mapped, nold, matcher, pi=None):
    weights = posterior(qr, mapped[0], nold, matcher, pi)
    pred = weights @ mapped[1]
    return {'mde': float(np.linalg.norm(pred-qx, axis=1).mean()),
            'anchor_weight_mean': float(weights[:, nold:].sum(1).mean()),
            'anchor_weight_over_half': float((weights[:, nold:].sum(1) > .5).mean())}


def minimal_checks():
    rng = np.random.default_rng(43)
    proto = rng.uniform(-105, -45, (9, 4)); qr = rng.uniform(-105, -45, (3, 4))
    qr[0, 1] = 100; xy = rng.normal(size=(9, 2)); matcher = (16., -75.)
    w = posterior(qr, proto, 6, matcher)
    assert np.allclose(w @ xy, tobit_match(qr, proto, xy, *matcher))
    # Copying either entire group preserves that group's prior measure and output.
    wp = posterior(qr, proto, 6, matcher, .1)
    copied_old = np.vstack((proto[:6], proto[:6], proto[6:]))
    copied_xy = np.vstack((xy[:6], xy[:6], xy[6:]))
    assert np.allclose(wp @ xy, posterior(qr, copied_old, 12, matcher, .1) @ copied_xy)
    copied_new = np.vstack((proto, proto[6:]))
    assert np.allclose(wp @ xy, posterior(qr, copied_new, 6, matcher, .1) @ np.vstack((xy, xy[6:])))
    assert np.allclose(posterior(qr, proto, 6, matcher, 0) @ xy, tobit_match(qr, proto[:6], xy[:6], *matcher))
    assert np.allclose(posterior(qr, proto, 6, matcher, 1) @ xy, tobit_match(qr, proto[6:], xy[6:], *matcher))
    # Equal energy yields requested group prior; actual Query evidence can override it.
    equal = np.repeat(proto[:1], 9, axis=0)
    assert np.allclose(posterior(qr, equal, 6, matcher, .1)[:, 6:].sum(1), .1)
    assert np.allclose(posterior(qr, proto[:6], 6, matcher, .1) @ xy[:6], tobit_match(qr, proto[:6], xy[:6], *matcher))
    context = (proto[:6], xy[:6], np.zeros((4, 0)), np.zeros(0))
    empty = four_maps(context, qr[:0], xy[:0], (60., 1., 0.))
    assert all(np.array_equal(value[0], proto[:6]) for value in empty.values())
    print('checks passed: frozen matcher, whole-group replication, endpoints, equal energy, empty Support', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checks-only', action='store_true'); args = parser.parse_args()
    minimal_checks()
    if args.checks_only:
        return
    args.output.mkdir(parents=True, exist_ok=False); start = time.time()
    prior_manifest = json.loads((PRIOR/'manifest.json').read_text())
    selections = json.loads((PRIOR/'selections.json').read_text())
    train, future = load_floors(RAW/'trainingData.csv'), load_floors(RAW/'validationData.csv')
    contexts = {f: old_context(train[f]['rssi'], train[f]['xy']) for f in future}
    episodes = {}
    for f, manifest in prior_manifest['floors'].items():
        ids = {int(z): i for i, z in enumerate(future[f]['row_ids'])}
        q = np.array([ids[z] for z in manifest['query_row_ids']])
        s = [np.array([ids[z] for z in order]) for order in manifest['draw_support_order']]
        assert all(not set(map(tuple, future[f]['xy'][q])) & set(map(tuple, future[f]['xy'][order])) for order in s)
        episodes[f] = (q, s)
    provenance = {'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'prior_manifest_sha256': hashlib.sha256((PRIOR/'manifest.json').read_bytes()).hexdigest(),
                  'prior_selections_sha256': hashlib.sha256((PRIOR/'selections.json').read_bytes()).hexdigest(),
                  'interpretation': 'Post-hoc follow-up of previously evaluated fixed Query, not a new blind benchmark.',
                  'floors': prior_manifest['floors']}
    (args.output/'manifest.json').write_text(json.dumps(provenance, indent=1))
    chosen_all, rows = {}, []
    for held in range(3):
        matcher = selections[str(held)]['matcher']; params = selections[str(held)]['params']['scm']
        source = [f for f in future if int(f[1]) != held]
        scores = {m: {str(pi): [] for pi in PIS} for m in ('group_append', 'group_field_append')}
        for f in source:
            v = future[f]; q, orders = episodes[f]; context = contexts[f]
            floor_scores = {m: {str(pi): [] for pi in PIS} for m in scores}
            for order in orders[:3]:
                for k in COUNTS:
                    s = order[:k]; candidates = four_maps(context, v['rssi'][s], v['xy'][s], params)
                    for mode in scores:
                        for pi in PIS:
                            floor_scores[mode][str(pi)].append(evaluate(v['rssi'][q], v['xy'][q], candidates[mode], len(context[0]), matcher, pi)['mde'])
            for mode in scores:
                for pi in PIS:
                    scores[mode][str(pi)].append(float(np.mean(floor_scores[mode][str(pi)])))
        chosen = {mode: min(PIS, key=lambda pi: np.mean(scores[mode][str(pi)])) for mode in scores}
        chosen_all[str(held)] = {'source_floors': source, 'matcher_frozen': matcher, 'gp_frozen': params,
                                'pi': chosen, 'source_equal_floor_grid': {m: {p: float(np.mean(z)) for p, z in score.items()} for m, score in scores.items()}}
        (args.output/'selections.json').write_text(json.dumps(chosen_all, indent=1))
        print('freeze', held, json.dumps(chosen_all[str(held)]['pi']), flush=True)
        for f in future:
            if int(f[1]) != held:
                continue
            v = future[f]; q, orders = episodes[f]; context = contexts[f]
            for draw, order in enumerate(orders):
                for k in COUNTS:
                    s = order[:k]; candidates = four_maps(context, v['rssi'][s], v['xy'][s], params)
                    for mode in METHODS:
                        pi = chosen[mode] if mode in chosen else None
                        row = {'floor': f, 'held_building': held, 'draw': draw, 'K': k, 'mode': mode,
                               'query_rows': len(q), 'query_positions': len(np.unique(v['xy'][q], axis=0)),
                               'old_candidates': len(context[0]), 'new_candidates': len(candidates[mode][0])-len(context[0]),
                               'prior_anchor_mass': pi if pi is not None else (len(candidates[mode][0])-len(context[0]))/len(candidates[mode][0]),
                               'pi': pi, **evaluate(v['rssi'][q], v['xy'][q], candidates[mode], len(context[0]), matcher, pi)}
                        rows.append(row)
                        with (args.output/'rows.jsonl').open('a') as stream:
                            stream.write(json.dumps(row)+'\n')
            print('floor', f, 'done', 'elapsed', round(time.time()-start, 1), flush=True)
    summary = {}
    for k in COUNTS:
        summary[str(k)] = {}
        for mode in METHODS:
            groups = [[r for r in rows if r['K']==k and r['mode']==mode and r['floor']==f] for f in future]
            summary[str(k)][mode] = {metric: float(np.mean([np.mean([r[metric] for r in group]) for group in groups])) for metric in ('mde', 'anchor_weight_mean', 'anchor_weight_over_half')}
    (args.output/'summary.json').write_text(json.dumps({'equal_floor': summary, 'elapsed_s': time.time()-start}, indent=1))
    print('DONE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
