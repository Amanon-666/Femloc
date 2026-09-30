"""观测模型自检：不做跨楼层迁移，用该层完整 trainingData 地图定位官方 validation。参数只按生成式估计。"""
import numpy as np
from data.uji import load_floors
from models.radio_map import cells, fit_observation, cell_predictive, loglik, posterior_mean
from scripts.evaluate_signal_calibrated_map import mde, wknn, lower_map
from scripts.evaluate_scm_tobit import tobit_match, PAIRS
R = "/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/"
tr, va = load_floors(R + "trainingData.csv"), load_floors(R + "validationData.csv")
SRC = sorted({f for p in PAIRS for f in p})
C = {f: cells(tr[f]) for f in SRC}
keys = ["old_quad", "old_tobit", "old_hurdle", "full_quad", "full_tobit", "full_hurdle"]
tab, fits = {}, []
for lo, up in PAIRS:
    obs = fit_observation([C[f] for f in SRC if f != up]); fits.append(obs)
    q, y = va[up]["rssi"], va[up]["xy"]
    for name, f, kq in (("old", lo, 1000), ("full", up, 8)):
        p, m, v, present = cell_predictive(C[f], obs)
        ll = loglik(q, p, m, v, present, obs["tau"])
        w = np.exp(ll - ll.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        tab.setdefault(up, {})[name + "_hurdle"] = mde(posterior_mean(ll, C[f]["xy"]), y)
        tab[up][name + "_neff"] = float(np.median(1 / (w ** 2).sum(1)))
        mp = lower_map(tr[f])
        tab[up][name + "_quad"] = mde(wknn(q, *mp, kq, 10), y)
        tab[up][name + "_tobit"] = mde(tobit_match(q, *mp, 16.0, -80.0), y)
o = fits[0]
print("obs fit (fold 1):", {k: round(v, 3) if isinstance(v, float) else v for k, v in o.items()})
print("across folds: sigma", [round(min(f["sigma"] for f in fits), 2), round(max(f["sigma"] for f in fits), 2)],
      "tau", [round(min(f["tau"] for f in fits), 2), round(max(f["tau"] for f in fits), 2)])
print("floor  " + " ".join(f"{k:>11s}" for k in keys) + "  neff_old neff_full")
for up, r in tab.items():
    print(f"{up}   " + " ".join(f"{r[k]:11.2f}" for k in keys) + f"  {r['old_neff']:8.2f} {r['full_neff']:9.2f}")
print("equal  " + " ".join(f"{np.mean([r[k] for r in tab.values()]):11.2f}" for k in keys))
