"""Same-floor temporal map maintenance; source-building-only hyperparameter selection.

LMC: k0 I + sum_q aq kq uq uq.T. Old-map PCA defines uq; this is a
testable contextual prior, not evidence that static and drift correlations agree.
Orthogonal uq permit exact small GP solves. diagonal_lmc retains all marginal
AP kernels/noise and removes only between-AP covariance.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve

from data.uji import load_floors
from scripts.evaluate_signal_calibrated_map import dbm, lookup as interpolated_lookup, position_map
from scripts.evaluate_scm_tobit import tobit_match

RAW = Path('/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc')
COUNTS = (3, 5, 10)
SPATIAL = [(l, n, 0.) for l in (20., 60., 120.) for n in (.1, .3, 1.)]
JOINT = [(l, n, g) for l in (20., 60.) for n in (.1, .3, 1.) for g in (.0, .3, 1., 3.)]


def lookup(xy, positions, prototypes, neighbors):
    result = interpolated_lookup(xy, positions, prototypes, neighbors)
    # Preserve measured old prototypes at exact old coordinates.
    hit = np.all(xy[:, None] == positions[None], axis=-1)
    rows, cols = np.where(hit)
    result[rows] = prototypes[cols]
    return result


def kernel(a, b, length):
    return np.exp(-np.square(a[:, None] - b[None]).sum(-1) / (2 * length**2))


def old_context(raw, xy):
    means, nodes, _ = position_map(raw, xy)
    centered = means - means.mean(0)
    # Compute PCA through small spatial Gram matrix, avoiding 520-dimensional SVD.
    vals, vecs = eigh(centered @ centered.T)
    ix = np.argsort(vals)[::-1][:8]
    ix = ix[vals[ix] > 1e-8]
    u = centered.T @ vecs[:, ix] / np.sqrt(vals[ix])[None]
    active = int(np.any(raw != 100, axis=0).sum())
    amplitudes = active * vals[ix] / vals[ix].sum()
    return means, nodes, u, amplitudes


def gp_update(to_xy, s_xy, residual, context, params, mode):
    if len(s_xy) == 0:
        return np.zeros((len(to_xy), residual.shape[1]))
    length, noise, gamma = params
    u, amps = context[2:]
    ks, kt = kernel(s_xy, s_xy, length), kernel(to_xy, s_xy, length)
    eye = np.eye(len(s_xy))
    base = kt @ solve(ks + noise * eye, residual, assume_a='pos')
    if gamma == 0 or mode == 'scm':
        return base
    # Different scales allow modes to have distinct spatial change patterns.
    scales = length * np.where(np.arange(u.shape[1]) < 2, 2., .5)
    sk = np.array([kernel(s_xy, s_xy, l) for l in scales])
    tk = np.array([kernel(to_xy, s_xy, l) for l in scales])
    a = gamma * amps
    if mode == 'lmc':
        z = residual @ u
        pred = np.column_stack([(kt + a[q] * tk[q]) @ solve(
            ks + a[q] * sk[q] + noise * eye, z[:, q], assume_a='pos')
            for q in range(u.shape[1])])
        return base + (pred - base @ u) @ u.T
    # Same marginal kernels as the LMC, AP by AP; only off-diagonal B removed.
    weights = u**2 * a[None]
    cov = ks[None] + np.einsum('aq,qij->aij', weights, sk) + noise * eye[None]
    cross = kt[None] + np.einsum('aq,qij->aij', weights, tk)
    alpha = np.linalg.solve(cov, residual.T[:, :, None])[:, :, 0]
    return np.einsum('aij,aj->ia', cross, alpha)


def split_floor(floor, seed):
    coords, inv = np.unique(floor['xy'], axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(coords))
    nq = min(max(1, int(np.ceil(.4 * len(coords)))), len(coords) - 10)
    query = np.flatnonzero(np.isin(inv, order[:nq]))
    pool = order[nq:]
    reps = np.array([rng.choice(np.flatnonzero(inv == p)) for p in pool])
    return query, reps, coords[pool]


def maps(context, raw, xy, params, mode):
    old, nodes = context[:2]
    if len(raw) == 0:
        return old, nodes
    anchor = dbm(raw)
    residual = anchor - lookup(xy, nodes, old, 3)
    if mode == 'old':
        return old, nodes
    if mode == 'anchors':
        return anchor, xy
    if mode == 'append':
        return np.vstack((old, anchor)), np.vstack((nodes, xy))
    if mode == 'global':
        virtual = old + residual.mean(0)
    else:
        virtual = old + gp_update(nodes, xy, residual, context, params, mode)
    return np.vstack((np.clip(virtual, -110, 0), anchor)), np.vstack((nodes, xy))


def error(raw, xy, mapped, matcher):
    pred = tobit_match(raw, *mapped, *matcher)
    return float(np.linalg.norm(pred - xy, axis=1).mean())


def reconstruction(context, sr, sx, qr, qx, params, mode, threshold):
    old, nodes = context[:2]
    prior = lookup(qx, nodes, old, 3)
    residual = dbm(sr) - lookup(sx, nodes, old, 3)
    if mode == 'old' or mode == 'append':
        pred = prior
    elif mode == 'global':
        pred = prior + residual.mean(0)
    elif mode == 'anchors':
        pred = lookup(qx, sx, dbm(sr), min(3, len(sx)))
    else:
        pred = prior + gp_update(qx, sx, residual, context, params, mode)
    pred = np.clip(pred, -110, 0)
    truth, visible = dbm(qr), qr != 100
    squared = (pred - truth)**2
    # AP equal-weight visible-only error; missing coding is separately reported.
    ap_count = visible.sum(0)
    ap_mse = (squared * visible).sum(0)[ap_count > 0] / ap_count[ap_count > 0]
    detecting = pred > threshold
    tp, fp, fn = (detecting & visible).sum(), (detecting & ~visible).sum(), (~detecting & visible).sum()
    return {'coded_rmse': float(np.sqrt(squared.mean())),
            'visible_rmse': float(np.sqrt(squared[visible].mean())),
            'ap_equal_visible_rmse': float(np.sqrt(ap_mse.mean())),
            'detection_f1': float(2 * tp / (2 * tp + fp + fn))}


def minimal_checks():
    rng = np.random.default_rng(7)
    nodes = rng.normal(size=(6, 2)); u, _ = np.linalg.qr(rng.normal(size=(4, 2)))
    context = (np.zeros((6, 4)), nodes, u, np.array([3., 1.]))
    sx = nodes[:3]; to = nodes[3:]; r = rng.normal(size=(3, 4)); par = (1., .3, 2.)
    full = gp_update(to, sx, r, context, par, 'lmc')
    ks, kt = kernel(sx, sx, 1.), kernel(to, sx, 1.)
    K, C = np.kron(ks, np.eye(4)), np.kron(kt, np.eye(4))
    for q, l in enumerate((2., 2.)):
        b = 2. * context[3][q] * np.outer(u[:, q], u[:, q])
        K += np.kron(kernel(sx, sx, l), b); C += np.kron(kernel(to, sx, l), b)
    brute = (C @ np.linalg.solve(K + .3 * np.eye(12), r.ravel())).reshape(3, 4)
    assert np.allclose(full, brute, atol=1e-9)
    diagonal = gp_update(to, sx, r, context, par, 'diagonal_lmc')
    Kd, Cd = np.kron(ks, np.eye(4)), np.kron(kt, np.eye(4))
    for q, l in enumerate((2., 2.)):
        b = np.diag(2. * context[3][q] * u[:, q]**2)
        Kd += np.kron(kernel(sx, sx, l), b); Cd += np.kron(kernel(to, sx, l), b)
    assert np.allclose(np.diag(K), np.diag(Kd))
    assert np.allclose(diagonal, (Cd @ np.linalg.solve(Kd + .3 * np.eye(12), r.ravel())).reshape(3, 4))
    assert np.allclose(gp_update(to, sx, r, context, (1., .3, 0.), 'lmc'), kt @ solve(ks + .3 * np.eye(3), r))
    assert np.allclose(gp_update(to, sx, np.zeros_like(r), context, par, 'lmc'), 0)
    assert np.allclose(full, gp_update(to + 100, sx + 100, r, context, par, 'lmc'))
    p = np.array([2, 0, 3, 1]); perm = (context[0][:, p], nodes, u[p], context[3])
    assert np.allclose(full[:, p], gp_update(to, sx, r[:, p], perm, par, 'lmc'))
    assert np.isfinite(gp_update(to, sx[:1], r[:1], context, par, 'lmc')).all()
    assert np.allclose(gp_update(to, sx[:0], r[:0], context, par, 'lmc'), 0)
    print('minimal_checks passed: brute full GP, zero residual, gamma=0, translation, AP permutation, one anchor', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checks-only', action='store_true'); args = parser.parse_args()
    minimal_checks()
    if args.checks_only:
        return
    start = time.time(); args.output.mkdir(parents=True, exist_ok=False)
    train, new = load_floors(RAW / 'trainingData.csv'), load_floors(RAW / 'validationData.csv')
    context = {f: old_context(train[f]['rssi'], train[f]['xy']) for f in new}
    partitions = {f: split_floor(v, 7300 + int(f[1])*100 + int(f[3])) for f, v in new.items()}
    manifest = {'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'floors': {}}
    for f, (q, pool, _) in partitions.items():
        manifest['floors'][f] = {'query_row_ids': new[f]['row_ids'][q].tolist(), 'support_pool_row_ids': new[f]['row_ids'][pool].tolist()}
        assert not set(map(tuple, new[f]['xy'][q])) & set(map(tuple, new[f]['xy'][pool]))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    selections = {}; rows = []
    for held in range(3):
        source = [f for f in new if int(f[1]) != held]
        # Common matcher chosen using old-map-only source localisation; fixed across all methods.
        matcher_grid = [(s, t) for s in (12., 16., 24.) for t in (-85., -75.)]
        matcher_scores = [np.mean([error(new[f]['rssi'][partitions[f][0]], new[f]['xy'][partitions[f][0]], context[f][:2], m) for f in source]) for m in matcher_grid]
        matcher = matcher_grid[int(np.argmin(matcher_scores))]
        candidates = [('scm', p) for p in SPATIAL] + [(m, p) for m in ('lmc', 'diagonal_lmc') for p in JOINT]
        scores = {str(c): [] for c in candidates}
        for f in source:
            v = new[f]; q, pool, _ = partitions[f]
            floor_scores = {str(c): [] for c in candidates}
            for draw in range(3):
                ordered = np.random.default_rng(9400 + draw).permutation(pool)
                for k in COUNTS:
                    s = ordered[:k]
                    for mode, params in candidates:
                        mapped = maps(context[f], v['rssi'][s], v['xy'][s], params, mode)
                        floor_scores[str((mode, params))].append(error(v['rssi'][q], v['xy'][q], mapped, matcher))
            for c in candidates:
                scores[str(c)].append(float(np.mean(floor_scores[str(c)])))
        chosen = {mode: min((p for m, p in candidates if m == mode), key=lambda p: np.mean(scores[str((mode, p))])) for mode in ('scm', 'lmc', 'diagonal_lmc')}
        selections[str(held)] = {'source_floors': source, 'matcher': matcher, 'params': chosen, 'source_score_grid': {c: float(np.mean(v)) for c, v in scores.items()}}
        print('freeze held', held, json.dumps({'matcher': matcher, 'params': chosen}), 'elapsed', round(time.time()-start, 1), flush=True)
        (args.output / 'selections.json').write_text(json.dumps(selections, indent=1))
        for f in new:
            if int(f[1]) != held:
                continue
            v = new[f]; q, pool, _ = partitions[f]
            qdistance = np.sqrt(np.square(v['xy'][q, None] - context[f][1][None]).sum(-1)).min(1)
            for draw in range(10):
                s_order = np.random.default_rng(10400 + draw).permutation(pool)
                manifest['floors'][f].setdefault('draw_support_order', []).append(v['row_ids'][s_order[:10]].tolist())
                for k in COUNTS:
                    s = s_order[:k]
                    modes = ('old', 'anchors', 'append', 'global', 'scm', 'lmc', 'diagonal_lmc', 'diagonal_best')
                    for mode in modes:
                        params = chosen['diagonal_lmc'] if mode == 'diagonal_best' else chosen['lmc'] if mode in ('lmc', 'diagonal_lmc') else chosen['scm']
                        actual_mode = 'diagonal_lmc' if mode == 'diagonal_best' else mode
                        mapped = maps(context[f], v['rssi'][s], v['xy'][s], params, actual_mode)
                        row = {'floor': f, 'held_building': held, 'draw': draw, 'K': k, 'mode': mode,
                               'mde': error(v['rssi'][q], v['xy'][q], mapped, matcher),
                               'query_rows': len(q), 'query_positions': len(np.unique(v['xy'][q], axis=0)),
                               'future_positions': len(np.unique(v['xy'], axis=0)), 'support_fraction': k / len(np.unique(v['xy'], axis=0)),
                               'old_1m_coverage': float(np.mean(qdistance <= 1)), 'old_nearest_m': float(qdistance.mean()),
                               **reconstruction(context[f], v['rssi'][s], v['xy'][s], v['rssi'][q], v['xy'][q], params, actual_mode, matcher[1])}
                        rows.append(row)
                        with (args.output / 'rows.jsonl').open('a') as stream:
                            stream.write(json.dumps(row) + '\n')
            print('target', f, 'done', 'elapsed', round(time.time()-start, 1), flush=True)
    summary = {}
    for k in COUNTS:
        summary[str(k)] = {}
        for mode in modes:
            grouped = [[r for r in rows if r['K'] == k and r['mode'] == mode and r['floor'] == f] for f in new]
            summary[str(k)][mode] = {metric: float(np.mean([np.mean([r[metric] for r in group]) for group in grouped]))
                                    for metric in ('mde', 'coded_rmse', 'visible_rmse', 'ap_equal_visible_rmse', 'detection_f1')}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    (args.output / 'summary.json').write_text(json.dumps({'equal_floor': summary, 'elapsed_s': time.time()-start}, indent=1))
    print('DONE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
