"""SCM v3：由单一生成模型推导的分层残差场 + 删失（Tobit）似然匹配。

模型（每 AP a、水平位置 p，单位 dBm）：
  v_a(p) = u_a(p) + b_a + g_a(p)
  b_a ~ N(0, τ²)：楼板穿透与垂直路径差，逐 AP 近似常数；
  g_a(·) ~ GP(0, σ_g²·RBF_ℓ)：墙体/几何残差，跨 AP 独立；
  锚点观测 s_i,a = v_a(y_i) + ε，ε ~ N(0, σ_s²)（每锚 3 扫描均值）；
  检出模型：AP a 被检出 ⟺ 场值高于设备阈值 θ（UJI 的 100 = 删失）。

推导结论（不是新变体，是同一模型的两个组成部分）：
1. 场估计 = b+g 的精确后验均值（σ_g²=1 参数化，ρ=τ²/σ_g²，λ=σ_s²/σ_g²）：
   v̂_a(p) = u_a(p) + (ρ·1 + k_p,D)ᵀ (ρ·11ᵀ + K_D + λI)⁻¹ r_a,D
   其中 D 是实际检出 AP a 的锚点集合（逐 AP 不同；未检出锚点的残差被删失，
   不得当作 −110 精确值——这正是 global_offset 变体失败的根源）。
   远离锚点（k_p→0）时退化为 u + 收缩的全局偏移 b̂_a：v1（尾部回退旧图，ρ→0）与
   global_offset（无空间项，σ_g²→0）是该式的两个极端，此前从未测试过混合式。
2. 匹配 = 候选位置上的对数似然（Tobit）：
   score(j) = −(1/2σ²)·Σ_{a∈D(x)}(x_a−V_ja)² + Σ_{a∉D(x)} logΦ((θ−V_ja)/σ)
   权重 w_j ∝ exp(score_j)（候选均匀先验下的后验），预测 = 后验均值坐标。
   第一项恰是"掩码平方距离"——推导解释了掩码变体为何更差：它丢掉了删失项，
   而"查询未检出但候选声称可见"的反对票带定位信息（与 AP-offset 分支结论互证）。

运行前登记的先验预测：P1 分层场 ≥ v1 场；P2 Tobit 匹配 ≥ 全距离二次匹配；
P3 增益集中在官方 validation 与 B1 类难转移。若 P1/P2 不成立，回到模型检查，不加补丁。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from data.uji import load_floors
from scripts.evaluate_signal_calibrated_map import (FLOOR, NO_SIGNAL, build_map, dbm,
                                                    episodes, lower_map, mde, position_map)
from scripts.evaluate_scm_v2 import PAIRS, SEED, EPISODES, QUERY_CAP, match as quad_match

NEIGHBORS = 3
KNN = 8
BETA_QUAD = 10
GRID_A = [(ell, lam, rho) for ell in (20.0, 40.0, 60.0, 100.0)
          for lam in (0.1, 0.3, 1.0) for rho in (3.0, 10.0, 30.0, 100.0)]
GRID_B = [(sigma, theta, topk) for sigma in (2.0, 4.0, 8.0)
          for theta in (-95.0, -90.0, -85.0) for topk in (None, 8)]
FROZEN_V1 = (60.0, 0.3)


def build_field(lower, s_rssi, s_xy, ell, lam, rho):
    """分层后验均值场。返回候选（场位置 ∪ 锚点）的 dBm 原型与坐标。"""
    protos, pos = lower
    x = dbm(s_rssi)
    det_scan = s_rssi != 100
    a_xy, a_inv = np.unique(s_xy, axis=0, return_inverse=True)
    a_inv = a_inv.ravel()
    K = len(a_xy)
    # 检出均值：每锚每 AP 只对实际检出的扫描平均（未检出是删失，不是 −110）
    s_clean = np.full((K, 520), NO_SIGNAL)
    for i in range(K):
        rows = a_inv == i
        d = det_scan[rows]
        n_det = d.sum(0)
        any_det = n_det > 0
        s_clean[i, any_det] = (x[rows][:, any_det] * d[:, any_det]).sum(0) / n_det[any_det]
    ulook_anchor = lookup_rows(a_xy, pos, protos)
    resid_full = s_clean - ulook_anchor
    det_any = s_clean > NO_SIGNAL + 0.5
    Kmat = np.exp(-((a_xy[:, None] - a_xy[None]) ** 2).sum(-1) / (2 * ell * ell))
    k_p = np.exp(-((pos[:, None] - a_xy[None]) ** 2).sum(-1) / (2 * ell * ell))
    V = protos.copy()
    for a in range(520):
        D = np.flatnonzero(det_any[:, a])
        if len(D) == 0:
            continue
        M = Kmat[np.ix_(D, D)] + lam * np.eye(len(D)) + rho
        w = np.linalg.solve(M, resid_full[D, a])
        V[:, a] += (rho + k_p[:, D]) @ w
    anchor_cand = np.where(det_any, s_clean, ulook_anchor)
    return np.vstack([V, anchor_cand]), np.vstack([pos, a_xy])


def lookup_rows(xy, pos, protos, neighbors=NEIGHBORS):
    d = np.sqrt(((xy[:, None] - pos[None]) ** 2).sum(-1))
    idx = np.argsort(d, 1)[:, :neighbors]
    w = 1 / (np.take_along_axis(d, idx, 1) + 1)
    w /= w.sum(1, keepdims=True)
    return (w[..., None] * protos[idx]).sum(1)


def tobit_match(qraw, protos, positions, sigma, theta, topk=None):
    det = (qraw != 100).astype(np.float64)
    x = dbm(qraw) * det
    q2 = (x * x).sum(1, keepdims=True)
    d2 = q2 + det @ (protos * protos).T - 2.0 * (x @ protos.T)
    L = np.clip(ndtr((theta - protos) / sigma), 1e-300, 1.0)
    score = -0.5 / sigma ** 2 * d2 + (1.0 - det) @ np.log(L).T
    score -= score.max(1, keepdims=True)
    w = np.exp(score)
    if topk is not None:
        idx = np.argsort(-w, 1)[:, :topk]
        ww = np.take_along_axis(w, idx, 1)
        ww /= ww.sum(1, keepdims=True)
        return (ww[..., None] * positions[idx]).sum(1)
    return (w / w.sum(1, keepdims=True)) @ positions


def episodes_data(train, n_eps):
    data = {}
    for b, f in PAIRS:
        old, new = lower_map(train[b]), train[f]
        rng = np.random.default_rng(SEED)
        eps = []
        for support, query in episodes(new["xy"], 10, n_eps, 3, rng):
            query = rng.choice(query, min(QUERY_CAP, len(query)), replace=False)
            eps.append((new["rssi"][support], new["xy"][support], new["rssi"][query], new["xy"][query]))
        data[(b, f)] = (old, new, eps)
    return data


def stage_a(data, grid, n_eps):
    """场参数选择：固定 v1 的二次匹配（k=8, β=10）以便与 v1 直接可比。"""
    scores = {}
    for combo in grid:
        per = {}
        for (b, f), (old, new, eps) in data.items():
            errs = []
            for s_rssi, s_xy, q_rssi, q_xy in eps:
                cand = build_field(old, s_rssi, s_xy, *combo)
                errs.append(mde(quad_match(q_rssi, *cand, KNN, BETA_QUAD, False), q_xy))
            per[f"{b}->{f}"] = errs
        scores[combo] = per
    equal = {c: float(np.mean([np.mean(v) for v in d.values()])) for c, d in scores.items()}
    return scores, equal


def stage_b(data, field_combo, grid):
    """匹配参数选择：给定场，比较 Tobit 与二次匹配。"""
    scores = {}
    for combo in grid:
        per = {}
        for (b, f), (old, new, eps) in data.items():
            errs = []
            for s_rssi, s_xy, q_rssi, q_xy in eps:
                cand = build_field(old, s_rssi, s_xy, *field_combo)
                sigma, theta, topk = combo
                errs.append(mde(tobit_match(q_rssi, *cand, sigma, theta, topk), q_xy))
            per[f"{b}->{f}"] = errs
        scores[combo] = per
    equal = {c: float(np.mean([np.mean(v) for v in d.values()])) for c, d in scores.items()}
    return scores, equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/signal_calibrated_map.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    train = load_floors(config["train_path"])
    validation = load_floors(config["validation_path"])
    report = {}

    print("stage A: field params (quad matching, k=8 beta=10)", flush=True)
    data = episodes_data(train, EPISODES)
    scores_a, equal_a = stage_a(data, GRID_A, EPISODES)
    best_a = min(equal_a, key=equal_a.get)
    # v1 参照（同 episode）：SCM v1 场 + 二次匹配
    per_v1 = {}
    for (b, f), (old, new, eps) in data.items():
        errs = []
        for s_rssi, s_xy, q_rssi, q_xy in eps:
            cand = build_map("scm", old, s_rssi, s_xy, FROZEN_V1, config)
            errs.append(mde(quad_match(q_rssi, *cand, KNN, BETA_QUAD, False), q_xy))
        per_v1[f"{b}->{f}"] = errs
    v1_equal = float(np.mean([np.mean(v) for v in per_v1.values()]))
    print("best field", best_a, round(equal_a[best_a], 3), "| v1 reference", round(v1_equal, 3), flush=True)
    report["stage_a"] = {"equal_weight": {str(c): v for c, v in equal_a.items()}, "best": str(best_a),
                         "v1_reference": v1_equal,
                         "lopo": lopo_audit(scores_a, GRID_A)}
    best_field = tuple(float(x) for x in best_a)

    print("stage B: matching params", flush=True)
    scores_b, equal_b = stage_b(data, best_field, GRID_B)
    best_b = min(equal_b, key=equal_b.get)
    print("best matching", best_b, round(equal_b[best_b], 3), flush=True)
    report["stage_b"] = {"equal_weight": {str(c): v for c, v in equal_b.items()}, "best": str(best_b),
                         "lopo": lopo_audit(scores_b, GRID_B)}

    print("phase 2: historical floors official validation", flush=True)
    sigma, theta, topk = best_b
    hist = {}
    for lower, upper in PAIRS:
        old, new, val = lower_map(train[lower]), train[upper], validation[upper]
        own = lower_map(new)
        rng = np.random.default_rng(SEED + 1000)
        errs = []
        for support, _ in episodes(new["xy"], 10, 20, 3, rng):
            cand = build_field(old, new["rssi"][support], new["xy"][support], *best_field)
            errs.append(mde(tobit_match(val["rssi"], *cand, sigma, theta, topk), val["xy"]))
        # v1 参照
        errs_v1 = []
        rng = np.random.default_rng(SEED + 1000)
        for support, _ in episodes(new["xy"], 10, 20, 3, rng):
            cand = build_map("scm", old, new["rssi"][support], new["xy"][support], FROZEN_V1, config)
            errs_v1.append(mde(quad_match(val["rssi"], *cand, KNN, BETA_QUAD, False), val["xy"]))
        hist[f"{lower}->{upper}"] = {
            "n": len(val["xy"]),
            "map_zero": mde(quad_match(val["rssi"], old[0], old[1], 1000, BETA_QUAD, False), val["xy"]),
            "scm_v1": float(np.mean(errs_v1)),
            "scm_v3": float(np.mean(errs)),
            "full_target_map": mde(quad_match(val["rssi"], own[0], own[1], KNN, BETA_QUAD, False), val["xy"])}
        print(" ", hist[f"{lower}->{upper}"], flush=True)
    report["historical_validation"] = hist

    print("phase 3: targets (exploratory appendix)", flush=True)
    targets = {}
    for name in config["targets"]:
        lower, upper = FLOOR[name]
        old, new, val = lower_map(train[lower]), train[upper], validation[upper]
        ids = {int(r): i for i, r in enumerate(new["row_ids"])}
        res = {"n_val": len(val["xy"]),
               "map_zero_val": mde(quad_match(val["rssi"], old[0], old[1], 1000, BETA_QUAD, False), val["xy"])}
        for seed_m in config["manifest_seeds"]:
            man = json.loads((Path(config["manifest_root"]) / f"seed_{seed_m}" / "manifest.json").read_text())[name]
            s = np.array([ids[int(r)] for r in man["support"]])
            q = np.array([ids[int(r)] for r in man["unseen_position_query"]])
            cand = build_field(old, new["rssi"][s], new["xy"][s], *best_field)
            res.setdefault("manifest_val", []).append(mde(tobit_match(val["rssi"], *cand, sigma, theta, topk), val["xy"]))
            res.setdefault("manifest_internal", []).append(mde(tobit_match(new["rssi"][q], *cand, sigma, theta, topk), new["xy"][q]))
            _, keys, inv = position_map(new["rssi"][s], new["xy"][s])
            sh = []
            for r_i in range(config["shuffle_repeats"]):
                s_xy2 = keys[np.random.default_rng(100 + r_i).permutation(len(keys))][inv]
                cand2 = build_field(old, new["rssi"][s], s_xy2, *best_field)
                sh.append(mde(tobit_match(val["rssi"], *cand2, sigma, theta, topk), val["xy"]))
            res.setdefault("manifest_val_shuffled", []).append(float(np.mean(sh)))
        targets[name] = res
        print(" ", name, {k: (round(float(np.mean(v)), 2) if isinstance(v, list) else round(v, 2))
                          for k, v in res.items()}, flush=True)
    report["targets"] = targets
    (args.output / "results.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("done", flush=True)


def lopo_audit(scores, grid):
    """留一转移：第 d 对的超参只由其余 6 对的均值选出，记录其在第 d 对的误差。"""
    keys = list(scores[next(iter(scores))].keys())
    out = {}
    for held in keys:
        pick = min(grid, key=lambda c: np.mean([np.mean(scores[c][k]) for k in keys if k != held]))
        out[held] = {"chosen": str(pick), "mde_m": float(np.mean(scores[pick][held]))}
    return out


if __name__ == "__main__":
    main()
