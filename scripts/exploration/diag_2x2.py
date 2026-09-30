"""诊断 2×2：{场: SCM 虚拟地图, LRM 后验均值} × {匹配: Tobit(σ=16,θ=−80), hurdle 观测模型}。
与 lrm_v2 同一批 episode（rng 1001）；只用于判断 LRM 落后于 SCM-T 的环节在场还是在匹配。"""
import numpy as np
from data.uji import load_floors
from models.radio_map import cells, fit_observation, posterior_mean
from models.cross_floor import fit_transfer, predictive, latent_loglik, nearest
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import build_map, episodes, lower_map, mde
R = "/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/"
tr, va = load_floors(R + "trainingData.csv"), load_floors(R + "validationData.csv")
floors = sorted({f for p in PAIRS for f in p})
C = {f: cells(tr[f]) for f in floors}
CFG = {"anchor_lookup_neighbors": 3}
sub = lambda fl, i: {k: fl[k][i] for k in ("rssi", "xy", "groups")}
res = {}
for lo, up in PAIRS:
    obs = fit_observation([C[f] for f in floors if f != up])
    tf = fit_transfer([(C[a], C[b]) for a, b in PAIRS if up not in (a, b)], obs)
    new, val, old = tr[up], va[up], lower_map(tr[lo])
    nL = len(old[1])
    for s, qi in episodes(new["xy"], 10, 20, 3, np.random.default_rng(1001)):
        ac = cells(sub(new, s))
        mu, v, pr, pos = predictive(C[lo], tf, obs, ac)
        sp, sxy = build_map("scm", old, new["rssi"][s], new["xy"][s], (60.0, 0.3), CFG)
        _, d = nearest(ac["xy"], old[1])
        rows = np.r_[np.arange(nL), nL + np.flatnonzero(d > 0.5)]
        assert np.allclose(sxy[rows], pos)
        for name, (q, y) in (("int", (new["rssi"][qi], new["xy"][qi])), ("val", (val["rssi"], val["xy"]))):
            out = {"scm_tobit": mde(tobit_match(q, sp, sxy, 16.0, -80.0), y),
                   "lrm_tobit": mde(tobit_match(q, np.clip(mu, -110, 0), pos, 16.0, -80.0), y)}
            for T in (1.0, 4.0):
                out[f"lrm_hurdle_T{T:g}"] = mde(posterior_mean(latent_loglik(q, mu, v, pr, tf, obs, False), pos, T), y)
                out[f"scm_hurdle_T{T:g}"] = mde(posterior_mean(latent_loglik(q, sp[rows], v, pr, tf, obs, False), pos, T), y)
            for k, e in out.items():
                res.setdefault((k, name), {}).setdefault(up, []).append(e)
    print(up, {f"{k}_{n}": round(np.mean(r[up]), 2) for (k, n), r in res.items()}, flush=True)
print("EQUAL-WEIGHT (7 floors)")
for k in ("scm_tobit", "lrm_tobit", "scm_hurdle_T1", "lrm_hurdle_T1", "scm_hurdle_T4", "lrm_hurdle_T4"):
    print(f"  {k:15s} int {np.mean([np.mean(v) for v in res[(k, 'int')].values()]):6.2f}"
          f"  val {np.mean([np.mean(v) for v in res[(k, 'val')].values()]):6.2f}")
print("done")
