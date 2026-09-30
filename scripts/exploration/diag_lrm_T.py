"""诊断（不改主线）：LRM 带锚点为何不如 SCM-T。
H1 似然过度自信：AP 条件独立的乘积使后验近乎 one-hot，需要温度 T（广义贝叶斯学习率）；
H2 空间相关长度被按 AP 去均值的矩估计压短（12 m），丢失中程趋势。
判断只看同日 trainingData 异位置 Query（int）；val 仅报告。episode 与 lrm_v2 相同（rng 1001），可配对。"""
import numpy as np
from data.uji import load_floors
from models.radio_map import cells, fit_observation, posterior_mean
from models.cross_floor import fit_transfer, predictive, latent_loglik
from scripts.evaluate_scm_tobit import PAIRS
from scripts.evaluate_signal_calibrated_map import episodes, mde
R = "/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/"
tr, va = load_floors(R + "trainingData.csv"), load_floors(R + "validationData.csv")
floors = sorted({f for p in PAIRS for f in p})
C = {f: cells(tr[f]) for f in floors}
TS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
ELLS = (None, 30.0, 60.0)
sub = lambda fl, i: {k: fl[k][i] for k in ("rssi", "xy", "groups")}
res, neff = {}, {}
for lo, up in PAIRS:
    obs = fit_observation([C[f] for f in floors if f != up])
    new, val = tr[up], va[up]
    eps = list(episodes(new["xy"], 10, 20, 3, np.random.default_rng(1001)))
    for L in ELLS:
        tf = fit_transfer([(C[a], C[b]) for a, b in PAIRS if up not in (a, b)], obs, ell=L)
        key = "fit" if L is None else int(L)
        if L is None:
            print(up, "fitted ell", tf["ell"], "sc2 %.1f nug %.1f" % (tf["sc2"], tf["nug"]),
                  "corr", np.round(tf["corr"], 2).tolist(), flush=True)
        for s, qi in eps:
            mu, v, pr, pos = predictive(C[lo], tf, obs, cells(sub(new, s)))
            for name, (q, y) in (("int", (new["rssi"][qi], new["xy"][qi])), ("val", (val["rssi"], val["xy"]))):
                ll = latent_loglik(q, mu, v, pr, tf, obs, exact=False)
                for T in TS:
                    res.setdefault((key, T, name), {}).setdefault(up, []).append(mde(posterior_mean(ll, pos, T), y))
                    if key == "fit" and name == "val":
                        w = np.exp((ll - ll.max(1, keepdims=True)) / T); w /= w.sum(1, keepdims=True)
                        neff.setdefault(T, []).append(float(np.median(1 / (w ** 2).sum(1))))
print("corr distance bins (m):", np.round(tf["corr_d"], 1).tolist())
eq = {k: np.mean([np.mean(v) for v in d.values()]) for k, d in res.items()}
print("median effective candidates (val, fitted ell):", {T: round(np.mean(v), 2) for T, v in neff.items()})
print("ell     T    int    val")
for L in ("fit", 30, 60):
    for T in TS:
        print(f"{L!s:5s} {T:4.1f} {eq[(L, T, 'int')]:6.2f} {eq[(L, T, 'val')]:6.2f}")
best = min(((L, T) for L in ("fit", 30, 60) for T in TS), key=lambda k: eq[(k[0], k[1], "int")])
print("best by int:", best, "int %.2f val %.2f" % (eq[(*best, "int")], eq[(*best, "val")]))
print("per-floor val at best:", {f: round(np.mean(v), 2) for f, v in res[(*best, "val")].items()})
print("per-floor int at best:", {f: round(np.mean(v), 2) for f, v in res[(*best, "int")].items()})
print("reference (lrm_v2, same episodes): SCM-T int 8.36 val 11.06 | LRM T=1 int 9.61 val 11.93")
print("done")
