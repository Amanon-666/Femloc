"""迁移先验校准地图（TPM）。一条主线；SCM-T 是它的先验取恒等映射（ρ=1、β=0、不建模检出）时的特例。

1. 观测（models/radio_map.py）：UJI 的 100 是删失；σ_t 与强信号检出率 h 由源楼层估计。
2. 跨楼层先验（models/cross_floor.py）：目标层潜在水平 = CEN + ρ_z(旧层检出均值 − CEN) + β_z + 残差
   （Kennedy–O'Hagan 自回归迁移），旧层未检出按删失处理；参数只在源楼层对上估计。零标注先验地图取观测空间的
   期望插补值 E[x_imp]。
3. 锚点校正：10 位置×3 扫描的残差（锚点插补均值 − 先验地图）经平面 RBF-GP 插值（与 SCM 相同的场）。
4. 匹配：删失（Tobit）似然下的后验均值坐标。
场 (ℓ, λ) 与匹配 (σ, θ) 只在 7 对源转移的 trainingData episode 上联合选择（与 SCM-T 同一随机流）。第 d 对的先验只用
不含该层的楼层对拟合（留一楼层）。官方 validation 只在参数冻结后评价；三个 FeMLoc 目标层是探索性附录。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.uji import load_floors
from models.cross_floor import fit_transfer, prior_map
from models.radio_map import cells, fit_observation
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import FLOOR, build_map, episodes, lower_map, mde, position_map

SEED, EPISODES, QUERY_CAP = 1, 20, 300
CFG = {"anchor_lookup_neighbors": 3}
SCM_T = ((60.0, 0.3), (16.0, -80.0))
FIELDS = [(ell, lam) for ell in (20.0, 40.0, 60.0, 100.0) for lam in (0.1, 0.3, 1.0, 3.0)]
MATCHES = [(sig, th) for sig in (8.0, 12.0, 16.0, 24.0) for th in (-90.0, -85.0, -80.0, -75.0)]
K_CURVE = (3, 5, 10, 20)


def transfer_prior(C, floors, held=None):
    """held 楼层及含它的楼层对都不参与观测常数和先验的估计。"""
    obs = fit_observation([C[f] for f in floors if f != held])
    return fit_transfer([(C[a], C[b]) for a, b in PAIRS if held not in (a, b)], obs)


def calibrate(base, rssi, xy, field):
    return build_map("scm", base, rssi, xy, field, CFG)


def select(tr, bases):
    grid = [(f, m) for f in FIELDS for m in MATCHES]
    per, ref = {g: {} for g in grid}, {}
    for lo, up in PAIRS:
        new, old = tr[up], lower_map(tr[lo])
        rng = np.random.default_rng(SEED)
        errs, ref_errs = {g: [] for g in grid}, []
        for s, q in episodes(new["xy"], 10, EPISODES, 3, rng):
            q = rng.choice(q, min(QUERY_CAP, len(q)), replace=False)
            qx, qy, sx, sy = new["rssi"][q], new["xy"][q], new["rssi"][s], new["xy"][s]
            for f in FIELDS:
                cand = calibrate(bases[up], sx, sy, f)
                for m in MATCHES:
                    errs[(f, m)].append(mde(tobit_match(qx, *cand, *m), qy))
            ref_errs.append(mde(tobit_match(qx, *calibrate(old, sx, sy, SCM_T[0]), *SCM_T[1]), qy))
        for g in grid:
            per[g][up] = float(np.mean(errs[g]))
        ref[up] = float(np.mean(ref_errs))
    equal = {g: float(np.mean(list(v.values()))) for g, v in per.items()}
    lopo = {}
    for held in ref:
        pick = min(grid, key=lambda g: np.mean([per[g][p] for p in ref if p != held]))
        lopo[held] = {"field": list(pick[0]), "match": list(pick[1]), "tpm_m": per[pick][held], "scm_t_m": ref[held]}
    return min(grid, key=equal.get), equal, lopo, ref


def historical(tr, va, lo, up, base, field, match):
    new, val, old = tr[up], va[up], lower_map(tr[lo])
    vq, vy = val["rssi"], val["xy"]
    row = {"n": len(vy),
           "old_zero": mde(tobit_match(vq, *old, *SCM_T[1]), vy),
           "prior_zero": mde(tobit_match(vq, *base, *match), vy),
           "full_map": mde(tobit_match(vq, *lower_map(new), *match), vy)}
    methods = {"scm_t": (old, *SCM_T), "tpm_v1": (base, *SCM_T), "tpm": (base, field, match)}
    rng = np.random.default_rng(SEED + 1000)
    for s, qi in episodes(new["xy"], 10, 20, 3, rng):
        sets = {"val": (vq, vy), "int": (new["rssi"][qi], new["xy"][qi])}
        for name, (b, f, m) in methods.items():
            cand = calibrate(b, new["rssi"][s], new["xy"][s], f)
            for split, (q, y) in sets.items():
                row.setdefault(f"{name}_{split}", []).append(mde(tobit_match(q, *cand, *m), y))
    rng = np.random.default_rng(SEED + 2000)
    for count in K_CURVE:
        for s, _ in episodes(new["xy"], count, 20, 3, rng):
            for name in ("scm_t", "tpm"):
                b, f, m = methods[name]
                row.setdefault(f"K{count}_{name}", []).append(
                    mde(tobit_match(vq, *calibrate(b, new["rssi"][s], new["xy"][s], f), *m), vy))
    return {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in row.items()}


def targets(tr, va, C, tf, config, field, match):
    out = {}
    for name in config["targets"]:
        lo = FLOOR[name][0]
        new, val, old, base = tr[name], va[name], lower_map(tr[lo]), prior_map(C[lo], tf)
        ids = {int(r): i for i, r in enumerate(new["row_ids"])}
        vq, vy = val["rssi"], val["xy"]
        row = {"old_zero_val": mde(tobit_match(vq, *old, *SCM_T[1]), vy),
               "prior_zero_val": mde(tobit_match(vq, *base, *match), vy)}
        for seed in config["manifest_seeds"]:
            man = json.loads((Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json").read_text())[name]
            s = np.array([ids[int(r)] for r in man["support"]])
            q = np.array([ids[int(r)] for r in man["unseen_position_query"]])
            sx, sy = new["rssi"][s], new["xy"][s]
            for mname, (b, f, m) in {"scm_t": (old, *SCM_T), "tpm": (base, field, match)}.items():
                cand = calibrate(b, sx, sy, f)
                for split, (qq, yy) in {"val": (vq, vy), "int": (new["rssi"][q], new["xy"][q])}.items():
                    row.setdefault(f"{mname}_{split}", []).append(mde(tobit_match(qq, *cand, *m), yy))
            # 机制诊断：30 条 RSSI 与 10 个坐标不变，只打乱"哪组扫描对应哪个坐标"
            _, keys, inv = position_map(sx, sy)
            shuffled = [mde(tobit_match(vq, *calibrate(base, sx, keys[np.random.default_rng(100 + r).permutation(len(keys))][inv],
                                                       field), *match), vy) for r in range(config["shuffle_repeats"])]
            row.setdefault("tpm_val_shuffled", []).append(float(np.mean(shuffled)))
        out[name] = {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in row.items()}
    return out


def summary(tf):
    return {"rho": tf["rho"].tolist(), "beta": tf["bbar"].tolist(), "s_b": np.sqrt(tf["sb2"]).tolist(),
            "s_w": float(np.sqrt(tf["s_w2"])), "theta": tf["theta"], "s": tf["s"], "h": tf["h"],
            "mu0": tf["mu0"].tolist(), "v_L0": tf["vL0"].tolist(), "sigma_total": tf["sigma_total"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/signal_calibrated_map.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    tr, va = load_floors(config["train_path"]), load_floors(config["validation_path"])
    floors = sorted({f for p in PAIRS for f in p})
    C = {f: cells(tr[f]) for f in floors}
    priors = {up: transfer_prior(C, floors, up) for _, up in PAIRS}
    bases = {up: prior_map(C[lo], priors[up]) for lo, up in PAIRS}

    (field, match), equal, lopo, ref = select(tr, bases)
    gains = [v["scm_t_m"] - v["tpm_m"] for v in lopo.values()]
    print(f"selected field {field} match {match}: source {equal[(field, match)]:.3f} | same params as SCM-T "
          f"{equal[SCM_T]:.3f} | SCM-T {np.mean(list(ref.values())):.3f}", flush=True)
    print(f"LOPO (params from the other 6 pairs) gain vs SCM-T {np.mean(gains):.3f} m, improved {sum(g > 0 for g in gains)}/7", flush=True)
    report = {"selected": {"field": list(field), "match": list(match)}, "lopo": lopo, "scm_t_source": ref,
              "source_equal_weight": {str(g): v for g, v in equal.items()},
              "priors": {up: summary(tf) for up, tf in priors.items()}}

    hist = {up: historical(tr, va, lo, up, bases[up], field, match) for lo, up in PAIRS}
    for up, row in hist.items():
        print(up, {k: round(v, 2) for k, v in row.items()}, flush=True)
    keys = [k for k in hist[PAIRS[0][1]] if k != "n"]
    report["historical"] = hist
    report["historical_equal"] = {k: float(np.mean([r[k] for r in hist.values()])) for k in keys}
    print("EQUAL", {k: round(v, 2) for k, v in report["historical_equal"].items()}, flush=True)
    for a, b in [("old_zero", "prior_zero"), ("scm_t_val", "tpm_val"), ("scm_t_int", "tpm_int"), ("tpm_v1_val", "tpm_val")] + \
            [(f"K{k}_scm_t", f"K{k}_tpm") for k in K_CURVE]:
        d = [hist[f][a] - hist[f][b] for f in hist]
        print(f"  {b} vs {a}: gain {np.mean(d):.2f} m, improved {sum(x > 0 for x in d)}/7, per floor {np.round(d, 2).tolist()}", flush=True)

    tf = transfer_prior(C, floors)
    report["target_prior"] = summary(tf)
    report["targets"] = targets(tr, va, C, tf, config, field, match)
    for name, row in report["targets"].items():
        print(name, {k: round(v, 2) for k, v in row.items()}, flush=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
