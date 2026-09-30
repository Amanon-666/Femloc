"""潜在无线电地图（LRM）主评价：同一观测模型贯穿先验拟合、锚点删失 EP 更新与匹配；严格留一楼层。

7 个历史上层楼层逐一留出：观测模型只用其余楼层，转移先验只用不含该层的楼层对；该层只读 10 位置×3 扫描
trainingData 锚点。两种观测形式都报告：M1（单潜变量，检出项以观测值为条件，由模型精确导出）与
M2（检出与数值在候选处条件独立）。二者取舍只看同日异位置 trainingData Query（与 SCM-T 选参同一数据权限），
官方 validation 只作评价。同一批锚点配对报告 SCM（v1 场 + 二次匹配）与 SCM-T（v1 场 + 删失匹配）。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.uji import load_floors
from models.cross_floor import fit_transfer, latent_loglik, predictive
from models.radio_map import cell_predictive, cells, fit_observation, loglik, posterior_mean
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import FLOOR, build_map, episodes, lower_map, mde, wknn

SEED = 1
FIELD, QUAD, ZERO_QUAD, TOBIT = (60.0, 0.3), (8, 10), (1000, 10), (16.0, -80.0)
CFG = {"anchor_lookup_neighbors": 3}
FORMS = {"m1": True, "m2": False}


def subset(floor, idx):
    return {k: floor[k][idx] for k in ("rssi", "xy", "groups")}


def add_lrm(acc, tag, pred, tf, obs, sets):
    mu, v, present, pos = pred
    for name, (q, y) in sets.items():
        for form, exact in FORMS.items():
            ll = latent_loglik(q, mu, v, present, tf, obs, exact)
            acc.setdefault(f"{tag}_{form}_{name}", []).append(mde(posterior_mean(ll, pos), y))


def add_support(acc, old, lc, tf, obs, new, s, sets):
    cand = build_map("scm", old, new["rssi"][s], new["xy"][s], FIELD, CFG)
    for name, (q, y) in sets.items():
        acc.setdefault(f"scm_quad_{name}", []).append(mde(wknn(q, *cand, *QUAD), y))
        acc.setdefault(f"scm_tobit_{name}", []).append(mde(tobit_match(q, *cand, *TOBIT), y))
    ac = cells(subset(new, s))
    add_lrm(acc, "lrm", predictive(lc, tf, obs, ac), tf, obs, sets)
    add_lrm(acc, "lrm_nocens", predictive(lc, tf, obs, ac, censored=False), tf, obs, sets)


def summary(acc):
    return {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in acc.items()}


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
    report = {"historical": {}, "priors": {}, "targets": {}}
    for lo, up in PAIRS:
        obs = fit_observation([C[f] for f in floors if f != up])
        tf = fit_transfer([(C[a], C[b]) for a, b in PAIRS if up not in (a, b)], obs)
        new, val, old = tr[up], va[up], lower_map(tr[lo])
        vq, vy = val["rssi"], val["xy"]
        acc = {"n": len(vy),
               "old_quad_val": mde(wknn(vq, *old, *ZERO_QUAD), vy),
               "old_tobit_val": mde(tobit_match(vq, *old, *TOBIT), vy),
               "full_quad_val": mde(wknn(vq, *lower_map(new), *QUAD), vy),
               "full_hurdle_val": mde(posterior_mean(loglik(vq, *cell_predictive(C[up], obs), obs["tau"]),
                                                     C[up]["xy"]), vy)}
        zero = predictive(C[lo], tf, obs)
        add_lrm(acc, "zero", zero, tf, obs, {"val": (vq, vy)})
        rng = np.random.default_rng(SEED + 1000)
        for s, qi in episodes(new["xy"], 10, 20, 3, rng):
            sets = {"val": (vq, vy), "int": (new["rssi"][qi], new["xy"][qi])}
            acc.setdefault("old_quad_int", []).append(mde(wknn(sets["int"][0], *old, *ZERO_QUAD), sets["int"][1]))
            add_lrm(acc, "zero", zero, tf, obs, {"int": sets["int"]})
            add_support(acc, old, C[lo], tf, obs, new, s, sets)
        report["historical"][up] = summary(acc)
        report["priors"][up] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in {**tf, **obs}.items()}
        print(up, {k: round(v, 2) for k, v in report["historical"][up].items()}, flush=True)
    keys = [k for k in report["historical"][PAIRS[0][1]] if k != "n"]
    eq = {k: float(np.mean([r[k] for r in report["historical"].values()])) for k in keys}
    report["historical_equal"] = eq
    choice = {tag: min(FORMS, key=lambda f: eq[f"{tag}_{f}_int"]) for tag in ("zero", "lrm")}
    report["form_choice"] = choice
    print("EQUAL", {k: round(v, 2) for k, v in eq.items()}, flush=True)
    print("form choice (same-day trainingData queries only):", choice, flush=True)

    obs = fit_observation([C[f] for f in floors])
    tf = fit_transfer([(C[a], C[b]) for a, b in PAIRS], obs)
    for name in config["targets"]:
        lo = FLOOR[name][0]
        new, val, old, lc = tr[name], va[name], lower_map(tr[FLOOR[name][0]]), cells(tr[lo])
        ids = {int(r): i for i, r in enumerate(new["row_ids"])}
        acc = {"old_quad_val": mde(wknn(val["rssi"], *old, *ZERO_QUAD), val["xy"]),
               "old_tobit_val": mde(tobit_match(val["rssi"], *old, *TOBIT), val["xy"])}
        add_lrm(acc, "zero", predictive(lc, tf, obs), tf, obs, {"val": (val["rssi"], val["xy"])})
        for seed in config["manifest_seeds"]:
            man = json.loads((Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json").read_text())[name]
            s = np.array([ids[int(r)] for r in man["support"]])
            q = np.array([ids[int(r)] for r in man["unseen_position_query"]])
            add_support(acc, old, lc, tf, obs, new, s,
                        {"val": (val["rssi"], val["xy"]), "int": (new["rssi"][q], new["xy"][q])})
        report["targets"][name] = summary(acc)
        print(name, {k: round(v, 2) for k, v in report["targets"][name].items()}, flush=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
