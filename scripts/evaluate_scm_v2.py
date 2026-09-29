"""SCM v2：支持集自适应（LOAO）超参、经验贝叶斯 AP 收缩与掩码距离匹配。

全局选择只用 7 对历史相邻楼层转移（trainingData，20 episode/对，seed 固定）；
官方 validation 只在冻结后评价；三个 FeMLoc 目标层只作探索性附录。复用 scm v1 原语。
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data.uji import load_floors
from scripts.evaluate_signal_calibrated_map import (FLOOR, NO_SIGNAL, dbm, position_map,
                                                    lookup, mde, episodes, lower_map)

PAIRS = [(f"B{b}F{f}", f"B{b}F{f + 1}") for b, top in ((0, 3), (1, 3), (2, 4)) for f in range(top - 1)]
LEN = (20.0, 40.0, 60.0, 100.0)
LAM = (0.1, 0.3, 1.0, 3.0)
KERNELS = ("rbf", "matern")
BETAS_M = (3, 10, 30, 100)
BETA_FULL = 10
KNN = 8
NEIGHBORS = 3
FROZEN = ("rbf", 60.0, 0.3)
EPISODES = 20
QUERY_CAP = 300
SEED = 1

SPECS = {
    "A_frozen": dict(loao=False, masked=False, weights="off", anchors=True),
    "B_frozen_masked": dict(loao=False, masked=True, weights="off", anchors=True),
    "C_loao_full": dict(loao=True, masked=False, weights="off", anchors=True),
    "D_loao_masked": dict(loao=True, masked=True, weights="off", anchors=True),
    "E_loao_masked_wiener": dict(loao=True, masked=True, weights="wiener", anchors=True),
    "F_loao_masked_hard": dict(loao=True, masked=True, weights="hard", anchors=True),
    "G_loao_masked_field": dict(loao=True, masked=True, weights="off", anchors=False),
}


def kern(a, b, ell, kernel):
    d2 = ((a[:, None] - b[None]) ** 2).sum(-1)
    if kernel == "rbf":
        return np.exp(-d2 / (2 * ell * ell))
    x = np.sqrt(5.0) * np.sqrt(d2) / ell
    return (1 + x + x * x / 3) * np.exp(-x)


def delta_field(positions, anchor_xy, resid, ell, lam, kernel):
    K = kern(anchor_xy, anchor_xy, ell, kernel)
    return kern(positions, anchor_xy, ell, kernel) @ np.linalg.solve(K + lam * np.eye(len(anchor_xy)), resid)


def match(qraw, protos, positions, k, beta, masked):
    """查询扫描对检索地图的 WKNN。两侧都在 clip((r+110)/110) 特征空间求距离；
    masked=True 时距离只在查询实际观测到的 AP 上求。"""
    f = np.clip((dbm(qraw) - NO_SIGNAL) / (-NO_SIGNAL), 0, 1)
    g = np.clip((protos - NO_SIGNAL) / (-NO_SIGNAL), 0, 1)
    if masked:
        m = (qraw != 100).astype(np.float64)
        c = np.maximum(m.sum(1, keepdims=True), 1.0)
        d = np.maximum((m * f * f).sum(1)[:, None] / c + (m @ (g * g).T) / c
                       - 2 * ((m * f) @ g.T) / c, 0)
    else:
        d = np.maximum((f * f).sum(1)[:, None] + (g * g).sum(1)[None] - 2 * f @ g.T, 0)
    k = min(k, len(protos))
    idx = np.argsort(d, 1)[:, :k]
    dd = np.take_along_axis(d, idx, 1)
    w = np.exp(-beta * (dd - dd[:, :1]))
    w /= w.sum(1, keepdims=True)
    return (w[..., None] * positions[idx]).sum(1)


def eb_weights(s_rssi, s_xy, lower, mode):
    """逐 AP 的经验贝叶斯收缩权重：空间方差远小于扫描噪声的 AP 残差被压低。只用 Support。"""
    protos, pos = lower
    x = dbm(s_rssi)
    r = x - lookup(s_xy, pos, protos, NEIGHBORS)
    _, _, inv = position_map(s_rssi, s_xy)
    anchor_mean = np.stack([r[inv == k].mean(0) for k in range(inv.max() + 1)])
    noise = (r - anchor_mean[inv]) ** 2
    noise = noise.sum(0) / max(len(x) - (inv.max() + 1), 1)
    sig = anchor_mean.var(0, ddof=1)
    if mode == "wiener":
        return sig / (sig + noise + 1e-9)
    return np.clip(1 - noise / np.maximum(sig, 1e-9), 0, 1)


def loao_pick(lower, s_rssi, s_xy, spec):
    """留一锚点代理：每个锚点轮流当伪 Query，只用其余锚点重建场；选出 (kernel, ell, lam, beta) 或 None。"""
    protos, pos = lower
    _, _, inv = position_map(s_rssi, s_xy)
    betas = BETAS_M if spec["masked"] else (BETA_FULL,)
    fields = [(ke, l_, lam) for ke in KERNELS for l_ in LEN for lam in LAM]
    cands = [(None, None, None, b) for b in betas] + [(f[0], f[1], f[2], b) for f in fields for b in betas]
    err = np.zeros(len(cands))
    for i in range(inv.max() + 1):
        rows = np.flatnonzero(inv == i)
        others = np.flatnonzero(inv != i)
        a_o, a_xy, _ = position_map(s_rssi[others], s_xy[others])
        r_o = a_o - lookup(a_xy, pos, protos, NEIGHBORS)
        w_o = eb_weights(s_rssi[others], s_xy[others], lower, spec["weights"]) if spec["weights"] != "off" else None
        base = (np.vstack([protos, a_o]), np.vstack([pos, a_xy])) if spec["anchors"] else (protos, pos)
        for ci, (ke, l_, lam, b) in enumerate(cands):
            if ke is None:
                mp = base
            else:
                d = delta_field(pos, a_xy, r_o, l_, lam, ke)
                if w_o is not None:
                    d = d * w_o
                v = np.clip(protos + d, NO_SIGNAL, 0)
                mp = (np.vstack([v, a_o]), np.vstack([pos, a_xy])) if spec["anchors"] else (v, pos)
            pred = match(s_rssi[rows], *mp, KNN, b, spec["masked"])
            err[ci] += np.linalg.norm(pred - s_xy[rows], axis=1).mean()
    return cands[int(np.argmin(err))]


def apply_support(lower, s_rssi, s_xy, spec, pick):
    """按选出的超参构造检索地图；返回 (protos, positions) 与匹配 beta。"""
    protos, pos = lower
    ke, l_, lam, b = pick
    anchor, anchor_xy, _ = position_map(s_rssi, s_xy)
    if ke is None:
        mp = (np.vstack([protos, anchor]), np.vstack([pos, anchor_xy])) if spec["anchors"] else (protos, pos)
        return mp, b
    r = anchor - lookup(anchor_xy, pos, protos, NEIGHBORS)
    d = delta_field(pos, anchor_xy, r, l_, lam, ke)
    if spec["weights"] != "off":
        d = d * eb_weights(s_rssi, s_xy, lower, spec["weights"])
    v = np.clip(protos + d, NO_SIGNAL, 0)
    mp = (np.vstack([v, anchor]), np.vstack([pos, anchor_xy])) if spec["anchors"] else (v, pos)
    return mp, b


def frozen_pick(spec, beta_masked):
    if spec["masked"]:
        return (*FROZEN, beta_masked)
    return (*FROZEN, BETA_FULL)


def eval_pair(task):
    pair, paths = task
    train = load_floors(paths["train"])
    lower, upper = pair
    old, new = lower_map(train[lower]), train[upper]
    rng = np.random.default_rng(SEED)
    rec = {name: [] for name in SPECS}
    rec["B_by_beta"] = {b: [] for b in BETAS_M}
    picks = {name: [] for name in SPECS if SPECS[name]["loao"]}
    for support, query in episodes(new["xy"], 10, EPISODES, 3, rng):
        query = rng.choice(query, min(QUERY_CAP, len(query)), replace=False)
        q_rssi, q_xy = new["rssi"][query], new["xy"][query]
        s_rssi, s_xy = new["rssi"][support], new["xy"][support]
        for beta in BETAS_M:
            mp, _ = apply_support(old, s_rssi, s_xy, SPECS["B_frozen_masked"], (*FROZEN, beta))
            rec["B_by_beta"][beta].append(mde(match(q_rssi, *mp, KNN, beta, True), q_xy))
        for name, spec in SPECS.items():
            if name == "B_frozen_masked":
                continue
            if name == "A_frozen":
                pick = frozen_pick(spec, None)
            else:
                pick = loao_pick(old, s_rssi, s_xy, spec)
                picks[name].append(pick)
            mp, beta = apply_support(old, s_rssi, s_xy, spec, pick)
            rec[name].append(mde(match(q_rssi, *mp, KNN, beta, spec["masked"]), q_xy))
    return pair, rec, picks


def equal_weight(per_pair):
    return float(np.mean([np.mean(v) for v in per_pair.values()]))


def run_phase1(paths, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(eval_pair, [(p, paths) for p in PAIRS]))
    rec = {p: r for p, r, _ in results}
    picks = {p: c for p, _, c in results}
    beta_masked = min(BETAS_M, key=lambda b: equal_weight({p: r["B_by_beta"][b] for p, r in rec.items()}))
    summary = {"B_frozen_masked": equal_weight({p: r["B_by_beta"][beta_masked] for p, r in rec.items()})}
    for name in SPECS:
        if name != "B_frozen_masked":
            summary[name] = equal_weight({p: r[name] for p, r in rec.items()})
    # LOPO 审计：第 d 对的方法只由其余 6 对的均值选出（B 依赖全局 beta，不参加）
    lopo = {}
    audited = [n for n in SPECS if n != "B_frozen_masked"]
    for pair, _ in rec.items():
        pick = min(audited, key=lambda n: equal_weight({p: r[n] for p, r in rec.items() if p != pair}))
        lopo[f"{pair[0]}->{pair[1]}"] = {"chosen": pick, "mde_m": float(np.mean(rec[pair][pick]))}
    return rec, picks, summary, beta_masked, lopo


def historical_val(train, validation, spec, beta_masked, draws=20):
    rng = np.random.default_rng(SEED + 1000)
    out = {}
    for lower, upper in PAIRS:
        old, new, val = lower_map(train[lower]), train[upper], validation[upper]
        own = lower_map(new)
        errs = []
        for support, _ in episodes(new["xy"], 10, draws, 3, rng):
            pick = loao_pick(old, new["rssi"][support], new["xy"][support], spec) if spec["loao"] \
                else frozen_pick(spec, beta_masked)
            mp, beta = apply_support(old, new["rssi"][support], new["xy"][support], spec, pick)
            errs.append(mde(match(val["rssi"], *mp, KNN, beta, spec["masked"]), val["xy"]))
        out[upper] = {"n": len(val["xy"]),
                      "map_zero": mde(match(val["rssi"], *old, 1000, BETA_FULL, False), val["xy"]),
                      "method": float(np.mean(errs)),
                      "full_target_map": mde(match(val["rssi"], *own, KNN, BETA_FULL, False), val["xy"])}
    return out


def target_eval(train, validation, spec, beta_masked, config):
    """三个 FeMLoc 目标层：固定 manifest + K 曲线 + 置乱诊断。只读 Support 的 RSSI 与坐标。"""
    out = {}
    rng = np.random.default_rng(config["k_curve_seed"])
    for name in config["targets"]:
        lower, upper = FLOOR[name]
        old, new, val = lower_map(train[lower]), train[upper], validation[name]
        ids = {int(r): i for i, r in enumerate(new["row_ids"])}
        res = {"n_val": len(val["xy"]),
               "map_zero_val": mde(match(val["rssi"], *old, 1000, BETA_FULL, False), val["xy"])}
        for seed_m in config["manifest_seeds"]:
            man = json.loads((Path(config["manifest_root"]) / f"seed_{seed_m}" / "manifest.json").read_text())[name]
            s = np.array([ids[int(r)] for r in man["support"]])
            q = np.array([ids[int(r)] for r in man["unseen_position_query"]])
            s_rssi, s_xy = new["rssi"][s], new["xy"][s]
            pick = loao_pick(old, s_rssi, s_xy, spec) if spec["loao"] else frozen_pick(spec, beta_masked)
            mp, beta = apply_support(old, s_rssi, s_xy, spec, pick)
            res.setdefault("manifest_val", []).append(mde(match(val["rssi"], *mp, KNN, beta, spec["masked"]), val["xy"]))
            res.setdefault("manifest_internal", []).append(mde(match(new["rssi"][q], *mp, KNN, beta, spec["masked"]), new["xy"][q]))
            _, keys, inv = position_map(s_rssi, s_xy)
            sh = []
            for r_i in range(config["shuffle_repeats"]):
                s_xy2 = keys[np.random.default_rng(100 + r_i).permutation(len(keys))][inv]
                pick2 = loao_pick(old, s_rssi, s_xy2, spec) if spec["loao"] else frozen_pick(spec, beta_masked)
                mp2, beta2 = apply_support(old, s_rssi, s_xy2, spec, pick2)
                sh.append(mde(match(val["rssi"], *mp2, KNN, beta2, spec["masked"]), val["xy"]))
            res.setdefault("manifest_val_shuffled", []).append(float(np.mean(sh)))
        for count in config["k_curve"]:
            for s, q in episodes(new["xy"], count, config["k_curve_draws"], config["scans_per_position"], rng):
                pick = loao_pick(old, new["rssi"][s], new["xy"][s], spec) if spec["loao"] else frozen_pick(spec, beta_masked)
                mp, beta = apply_support(old, new["rssi"][s], new["xy"][s], spec, pick)
                res.setdefault(f"K{count}_val", []).append(mde(match(val["rssi"], *mp, KNN, beta, spec["masked"]), val["xy"]))
                res.setdefault(f"K{count}_internal", []).append(mde(match(new["rssi"][q], *mp, KNN, beta, spec["masked"]), new["xy"][q]))
        out[name] = res
        print(" ", name, {k: (round(float(np.mean(v)), 2) if isinstance(v, list) else v)
                          for k, v in res.items() if not k.startswith("K")}, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/signal_calibrated_map.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-targets", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    paths = {"train": config["train_path"], "validation": config["validation_path"]}
    args.output.mkdir(parents=True, exist_ok=False)
    print("phase 1: source pairs", flush=True)
    rec, picks, summary, beta_masked, lopo = run_phase1(paths, args.workers)
    print("beta_masked", beta_masked)
    for k, v in sorted(summary.items(), key=lambda x: x[1]):
        print(f"  {k:24s} {v:.3f}")
    print("LOPO", json.dumps(lopo), flush=True)
    audited = [n for n in SPECS if n != "B_frozen_masked"]
    winner = min(audited, key=lambda n: summary[n])
    print("winner", winner, flush=True)
    train = load_floors(paths["train"])
    validation = load_floors(paths["validation"])
    spec = SPECS[winner]
    print("phase 2: historical floors official validation", flush=True)
    hist = {n: historical_val(train, validation, SPECS[n], beta_masked) for n in ("A_frozen", winner)}
    for n, h in hist.items():
        for p, v in h.items():
            print(" ", n, p, {k: round(x, 2) for k, x in v.items()})
        print(" ", n, "equal-weight method-vs-map:",
              round(equal_weight({p: v["method"] for p, v in h.items()}), 2),
              "vs", round(equal_weight({p: v["map_zero"] for p, v in h.items()}), 2), flush=True)
    pick_stats = {}
    for name in SPECS:
        if not SPECS[name]["loao"]:
            continue
        c = Counter(("none" if p[0] is None else f"{p[0]}/ell{p[1]}/b{p[3]}") for ps in picks.values() for p in ps[name])
        pick_stats[name] = c.most_common(8)
    report = {"summary_source_pairs": summary, "beta_masked": beta_masked, "winner": winner,
              "lopo_audit": lopo, "loao_pick_stats": pick_stats,
              "per_pair": {f"{p[0]}->{p[1]}": {n: [float(x) for x in r[n]] for n in SPECS if n != "B_frozen_masked"}
                           for p, r in rec.items()},
              "historical_validation": hist}
    if not args.skip_targets:
        print("phase 3: targets (exploratory)", flush=True)
        report["targets"] = target_eval(train, validation, spec, beta_masked, config)
    (args.output / "results.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
