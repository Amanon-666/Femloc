"""结构检验：Kennedy–O'Hagan 自回归系数 ρ_z。留一楼层；ρ≡1 与 ρ_z 自由配对比较：
(1) 留出楼层对的删失复合似然（每单元，越低越好）；(2) 与 lrm_v2 相同 episode 的定位；T 只按 int 选，val 只报告。"""
import numpy as np
from data.uji import load_floors
from models.radio_map import cells, fit_observation, posterior_mean
from models.cross_floor import fit_transfer, latent_loglik, link_nll, predictive
from scripts.evaluate_scm_tobit import PAIRS
from scripts.evaluate_signal_calibrated_map import episodes, mde

R = "/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/"
tr, va = load_floors(R + "trainingData.csv"), load_floors(R + "validationData.csv")
floors = sorted({f for p in PAIRS for f in p})
C = {f: cells(tr[f]) for f in floors}
TS, TAGS = (1.0, 2.0, 3.0, 4.0, 6.0), (("rho1", False), ("rhoz", True))
sub = lambda fl, i: {k: fl[k][i] for k in ("rssi", "xy", "groups")}
res, held = {}, {t: {} for t, _ in TAGS}


def score(tag, kind, name, q, y, pred, tf, obs):
    mu, v, pr, pos = pred
    ll = latent_loglik(q, mu, v, pr, tf, obs, False)
    for T in TS:
        res.setdefault((tag, kind, name, T), {}).setdefault(up, []).append(mde(posterior_mean(ll, pos, T), y))


for lo, up in PAIRS:
    obs = fit_observation([C[f] for f in floors if f != up])
    train = [(C[a], C[b]) for a, b in PAIRS if up not in (a, b)]
    new, val = tr[up], va[up]
    eps = list(episodes(new["xy"], 10, 20, 3, np.random.default_rng(1001)))
    for tag, slope in TAGS:
        tf = fit_transfer(train, obs, slope=slope)
        held[tag][up] = link_nll([(C[lo], C[up])], tf, obs)
        print(f"{up} {tag}: held link nll {held[tag][up]:.4f} | train {tf['link_nll']:.4f} | rho {np.round(tf['rho'], 2).tolist()} "
              f"beta {np.round(tf['bbar'], 1).tolist()} theta {tf['theta']:.1f} s {tf['s']:.1f} ell {tf['ell']:.0f}", flush=True)
        zero = predictive(C[lo], tf, obs)
        score(tag, "zero", "val", val["rssi"], val["xy"], zero, tf, obs)
        for s, qi in eps:
            score(tag, "zero", "int", new["rssi"][qi], new["xy"][qi], zero, tf, obs)
            pred = predictive(C[lo], tf, obs, cells(sub(new, s)))
            score(tag, "lrm", "int", new["rssi"][qi], new["xy"][qi], pred, tf, obs)
            score(tag, "lrm", "val", val["rssi"], val["xy"], pred, tf, obs)

eq = lambda key: np.mean([np.mean(v) for v in res[key].values()])
print("held-out link nll (mean of 7 folds):", {t: round(float(np.mean(list(h.values()))), 4) for t, h in held.items()})
print("per fold rhoz - rho1:", {f: round(held["rhoz"][f] - held["rho1"][f], 4) for f in held["rho1"]})
for tag, _ in TAGS:
    for kind in ("zero", "lrm"):
        row = "  ".join(f"T{T:g} {eq((tag, kind, 'int', T)):.2f}/{eq((tag, kind, 'val', T)):.2f}" for T in TS)
        Tb = min(TS, key=lambda T: eq((tag, kind, "int", T)))
        print(f"{tag} {kind:4s} int/val  {row}  | best T by int {Tb:g}")
        print(f"     per-floor val at T{Tb:g}:", {f: round(float(np.mean(v)), 2) for f, v in res[(tag, kind, 'val', Tb)].items()})
print("reference (same episodes): SCM-T int 8.36 val 11.06 | old-map Tobit val 17.34")
print("done")
