"""按 docs/research/ICL_CONTEXT_DERIVATION.md 冻结的协议评价：TPM 先验作 TabICLv2 合成上下文。

每个 episode 的 Support 与 TPM 主线相同（K10: rng 1001；K 曲线: rng 2001，按 K 顺序消耗同一随机流），
同一次上下文拟合同时预测官方 validation 与同日异位置 Query。没有任何参数按成绩选择。
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from data.uji import load_floors
from models.cross_floor import prior_map
from models.icl_context import ContextRegressor, sample_prior_scans
from models.radio_map import cells
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import FLOOR, episodes, mde
from scripts.evaluate_transfer_map import calibrate, transfer_prior

FIELD, MATCH = (40.0, 0.3), (12.0, -85.0)
K_CURVE = (3, 5, 10, 20)
K10_METHODS = ("icl_anchors", "icl_prior", "icl_prior_anchors", "icl_prior_anchors_noflag", "icl_old_anchors")
CURVE_METHODS = ("icl_anchors", "icl_prior_anchors", "icl_old_anchors")
FLOOR_ID = {f: i for i, f in enumerate(["B0F1", "B0F2", "B1F1", "B1F2", "B2F1", "B2F2", "B2F3", "B0F3", "B1F3", "B2F4"])}


def old_scans(floor, rng, scans=3):
    """旧层真实扫描：去重后每个坐标最多 scans 条。"""
    _, first = np.unique(floor["groups"], return_index=True)
    xy = floor["xy"][first]
    keys, inv = np.unique(xy, axis=0, return_inverse=True)
    pick = np.concatenate([rng.choice(np.flatnonzero(inv.ravel() == k), min(scans, (inv.ravel() == k).sum()), replace=False)
                           for k in range(len(keys))])
    return floor["rssi"][first][pick], xy[pick]


def context(method, lc, tf, old_floor, s_rssi, s_xy, rng):
    """返回 (上下文 RSSI, 坐标, 来源标记或 None)。"""
    if method == "icl_anchors":
        return s_rssi, s_xy, None
    if method == "icl_old_anchors":
        r, xy = old_scans(old_floor, rng)
    else:
        r, xy = sample_prior_scans(lc, tf, rng)
    if method == "icl_prior":
        return r, xy, None
    ctx, cxy = np.vstack([r, s_rssi]), np.vstack([xy, s_xy])
    flag = None if method == "icl_prior_anchors_noflag" else np.r_[np.zeros(len(r)), np.ones(len(s_rssi))]
    return ctx, cxy, flag


def run_episode(reg, methods, key, lc, tf, old_floor, base, s_rssi, s_xy, sets, origin):
    out = {}
    cand = calibrate(base, s_rssi, s_xy, FIELD)
    for name, (q, y) in sets.items():
        out[f"tpm_{name}"] = mde(tobit_match(q, *cand, *MATCH), y)
    for m_id, m in enumerate(methods):
        rng = np.random.default_rng([*key, m_id])
        ctx, cxy, flag = context(m, lc, tf, old_floor, s_rssi, s_xy, rng)
        seed = int(np.random.default_rng([*key, m_id, 7]).integers(2 ** 30))
        pred, nfeat = reg.predict(ctx, cxy, {k: q for k, (q, _) in sets.items()}, origin, flag, seed)
        for name, (_, y) in sets.items():
            out[f"{m}_{name}"] = mde(pred[name], y)
        out[f"{m}_features"] = nfeat
    return out


def historical_floor(tr, va, C, floors, lo, up, device):
    tf = transfer_prior(C, floors, up)
    base = prior_map(C[lo], tf)
    new, val = tr[up], va[up]
    origin = C[lo]["xy"].mean(0)
    reg = ContextRegressor(device)
    fid = FLOOR_ID[up]
    rows = {"K10": [], "curve": {}}
    rng = np.random.default_rng(1001)
    for ep, (s, qi) in enumerate(episodes(new["xy"], 10, 20, 3, rng)):
        sets = {"val": (val["rssi"], val["xy"]), "int": (new["rssi"][qi], new["xy"][qi])}
        rows["K10"].append(run_episode(reg, K10_METHODS, (fid, 10, ep), C[lo], tf, tr[lo], base,
                                       new["rssi"][s], new["xy"][s], sets, origin))
        print(up, "K10", ep, {k: round(v, 2) for k, v in rows["K10"][-1].items() if not k.endswith("features")}, flush=True)
    rng = np.random.default_rng(2001)
    for count in K_CURVE:
        rows["curve"][count] = []
        for ep, (s, _) in enumerate(episodes(new["xy"], count, 20, 3, rng)):
            rows["curve"][count].append(run_episode(reg, CURVE_METHODS, (fid, count, 100 + ep), C[lo], tf, tr[lo], base,
                                                    new["rssi"][s], new["xy"][s], {"val": (val["rssi"], val["xy"])}, origin))
        print(up, "K", count, {k: round(float(np.mean([r[k] for r in rows["curve"][count]])), 2)
                                for k in rows["curve"][count][0] if not k.endswith("features")}, flush=True)
    return rows


def target_floor(tr, va, C, floors, name, config, device):
    lo = FLOOR[name][0]
    tf = transfer_prior(C, floors)
    base = prior_map(C[lo], tf)
    new, val = tr[name], va[name]
    ids = {int(r): i for i, r in enumerate(new["row_ids"])}
    origin = C[lo]["xy"].mean(0)
    reg = ContextRegressor(device)
    rows = []
    for seed in config["manifest_seeds"]:
        man = json.loads((Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json").read_text())[name]
        s = np.array([ids[int(r)] for r in man["support"]])
        q = np.array([ids[int(r)] for r in man["unseen_position_query"]])
        sets = {"val": (val["rssi"], val["xy"]), "int": (new["rssi"][q], new["xy"][q])}
        rows.append(run_episode(reg, K10_METHODS, (FLOOR_ID[name], 10, 200 + seed), C[lo], tf, tr[lo], base,
                                new["rssi"][s], new["xy"][s], sets, origin))
        print(name, seed, {k: round(v, 2) for k, v in rows[-1].items() if not k.endswith("features")}, flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/signal_calibrated_map.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--floors", nargs="+", required=True, help="历史上层楼层名，或 targets 表示附录三层")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    tr, va = load_floors(config["train_path"]), load_floors(config["validation_path"])
    floors = sorted({f for pair in PAIRS for f in pair})
    C = {f: cells(tr[f]) for f in floors}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    lower = {up: lo for lo, up in PAIRS}
    for f in args.floors:
        if f == "targets":
            for name in config["targets"]:
                C[FLOOR[name][0]] = cells(tr[FLOOR[name][0]])
                res = target_floor(tr, va, C, floors, name, config, args.device)
                (args.output / f"target_{name}.json").write_text(json.dumps({"commit": commit, "rows": res}, indent=1) + "\n")
        else:
            res = historical_floor(tr, va, C, floors, lower[f], f, args.device)
            (args.output / f"historical_{f}.json").write_text(json.dumps({"commit": commit, **res}, indent=1) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
