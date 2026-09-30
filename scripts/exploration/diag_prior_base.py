"""检验：SCM 的基图由旧层原图换成 LRM 零标注先验图（期望插补 dBm）；锚点 GP（ℓ=60, λ=0.3）与 Tobit（σ=16, θ=−80）不变。
E[x_imp] = h·(Φ(z)·μ + v·φ(z)/√(s²+v)) + (1 − h·Φ(z))·(−110)，z = (μ−θ)/√(s²+v)。严格留一楼层；与 SCM-T 同一批 episode。"""
import numpy as np
from scipy.special import ndtr
from data.uji import load_floors
from models.radio_map import cells, fit_observation
from models.cross_floor import fit_transfer, predictive
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import NO_SIGNAL, build_map, episodes, lower_map, mde

R = "/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/"
tr, va = load_floors(R + "trainingData.csv"), load_floors(R + "validationData.csv")
floors = sorted({f for p in PAIRS for f in p})
C = {f: cells(tr[f]) for f in floors}
CFG, FIELD, TOBIT = {"anchor_lookup_neighbors": 3}, (60.0, 0.3), (16.0, -80.0)


def prior_map(lc, tf, obs):
    mu, v, present, pos = predictive(lc, tf, obs)
    sd = np.sqrt(tf["s"] ** 2 + v)
    z = (mu - tf["theta"]) / sd
    Phi, phi = ndtr(z), np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    m = tf["h"] * (Phi * mu + v * phi / sd) + (1 - tf["h"] * Phi) * NO_SIGNAL
    m[:, ~present] = NO_SIGNAL
    return np.clip(m, NO_SIGNAL, 0), pos


rows = {}
for lo, up in PAIRS:
    obs = fit_observation([C[f] for f in floors if f != up])
    tf = fit_transfer([(C[a], C[b]) for a, b in PAIRS if up not in (a, b)], obs, slope=True)
    new, val, old = tr[up], va[up], lower_map(tr[lo])
    pri = prior_map(C[lo], tf, obs)
    assert np.allclose(pri[1], old[1])
    vq, vy = val["rssi"], val["xy"]
    r = {"old_zero_val": mde(tobit_match(vq, *old, *TOBIT), vy), "prior_zero_val": mde(tobit_match(vq, *pri, *TOBIT), vy)}
    for s, qi in episodes(new["xy"], 10, 20, 3, np.random.default_rng(1001)):
        iq, iy = new["rssi"][qi], new["xy"][qi]
        r.setdefault("old_zero_int", []).append(mde(tobit_match(iq, *old, *TOBIT), iy))
        r.setdefault("prior_zero_int", []).append(mde(tobit_match(iq, *pri, *TOBIT), iy))
        for tag, base in (("old", old), ("prior", pri)):
            cand = build_map("scm", base, new["rssi"][s], new["xy"][s], FIELD, CFG)
            r.setdefault(f"{tag}_scm_val", []).append(mde(tobit_match(vq, *cand, *TOBIT), vy))
            r.setdefault(f"{tag}_scm_int", []).append(mde(tobit_match(iq, *cand, *TOBIT), iy))
    rng = np.random.default_rng(2001)
    for K in (3, 5, 10, 20):
        for s, _ in episodes(new["xy"], K, 20, 3, rng):
            for tag, base in (("old", old), ("prior", pri)):
                cand = build_map("scm", base, new["rssi"][s], new["xy"][s], FIELD, CFG)
                r.setdefault(f"K{K}_{tag}_val", []).append(mde(tobit_match(vq, *cand, *TOBIT), vy))
    rows[up] = {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in r.items()}
    print(up, {k: round(v, 2) for k, v in rows[up].items()}, flush=True)
keys = list(rows[PAIRS[0][1]])
eq = {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}
print("EQUAL", {k: round(v, 2) for k, v in eq.items()})
for a, b in (("old_zero_val", "prior_zero_val"), ("old_zero_int", "prior_zero_int"), ("old_scm_val", "prior_scm_val"),
             ("old_scm_int", "prior_scm_int")) + tuple((f"K{K}_old_val", f"K{K}_prior_val") for K in (3, 5, 10, 20)):
    d = [rows[f][a] - rows[f][b] for f in rows]
    print(f"  {b:16s} vs {a:14s}: gain {np.mean(d):5.2f} m, improved {sum(x > 0 for x in d)}/7, per-floor {np.round(d, 2).tolist()}")
print("reference: SCM-T val 11.06 int 8.36; K curve val 14.60/13.01/11.06/9.73; LRM zero-shot hurdle T3 val 13.40")
print("done")
