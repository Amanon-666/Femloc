"""结构检验：锚点更新放在潜在空间（删失 EP 后验 → 期望插补图 E[x_imp]）还是插补空间（SCM GP，ℓ=60, λ=0.3）。
基图同为 LRM 零标注先验（ρ_z，严格留一楼层）；匹配同为 Tobit(16, −80)；episode 与 SCM-T 相同（K=10: rng 1001；K 曲线: rng 2001）。
2×2 诊断里的 lrm_tobit 把潜在后验均值直接当插补图用，量纲不一致；本检验两边都用期望插补值 E[x_imp]。"""
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
sub = lambda fl, i: {k: fl[k][i] for k in ("rssi", "xy", "groups")}


def ximp(mu, v, present, tf):
    """E[x_imp] = h(Φ(z)μ + vφ(z)/√(s²+v)) + (1 − hΦ(z))·(−110)，z = (μ−θ)/√(s²+v)。"""
    sd = np.sqrt(tf["s"] ** 2 + v)
    z = (mu - tf["theta"]) / sd
    Phi, phi = ndtr(z), np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    m = tf["h"] * (Phi * mu + v * phi / sd) + (1 - tf["h"] * Phi) * NO_SIGNAL
    m[:, ~present] = NO_SIGNAL
    return np.clip(m, NO_SIGNAL, 0)


rows = {}
for lo, up in PAIRS:
    obs = fit_observation([C[f] for f in floors if f != up])
    tf = fit_transfer([(C[a], C[b]) for a, b in PAIRS if up not in (a, b)], obs, slope=True)
    new, val, old = tr[up], va[up], lower_map(tr[lo])
    m0, v0, p0, pos0 = predictive(C[lo], tf, obs)
    prior = (ximp(m0, v0, p0, tf), pos0)
    vq, vy = val["rssi"], val["xy"]
    r = {}

    def run(tag, s, sets):
        maps = {"old_scm": build_map("scm", old, new["rssi"][s], new["xy"][s], FIELD, CFG),
                "prior_scm": build_map("scm", prior, new["rssi"][s], new["xy"][s], FIELD, CFG)}
        mu, v, pr, pos = predictive(C[lo], tf, obs, cells(sub(new, s)))
        maps["ep"] = (ximp(mu, v, pr, tf), pos)
        for m, cand in maps.items():
            for name, (q, y) in sets.items():
                r.setdefault(f"{tag}{m}_{name}", []).append(mde(tobit_match(q, *cand, *TOBIT), y))

    for s, qi in episodes(new["xy"], 10, 20, 3, np.random.default_rng(1001)):
        run("", s, {"val": (vq, vy), "int": (new["rssi"][qi], new["xy"][qi])})
    rng = np.random.default_rng(2001)
    for K in (3, 5, 10, 20):
        for s, _ in episodes(new["xy"], K, 20, 3, rng):
            run(f"K{K}_", s, {"val": (vq, vy)})
    rows[up] = {k: float(np.mean(x)) for k, x in r.items()}
    print(up, {k: round(x, 2) for k, x in rows[up].items() if not k.startswith("K")}, flush=True)

keys = list(rows[PAIRS[0][1]])
eq = {k: float(np.mean([rw[k] for rw in rows.values()])) for k in keys}
print("EQUAL", {k: round(x, 2) for k, x in eq.items()})
for base in ("prior_scm", "old_scm"):
    for tag in ("", "K3_", "K5_", "K10_", "K20_"):
        for name in (("val", "int") if tag == "" else ("val",)):
            a, b = f"{tag}{base}_{name}", f"{tag}ep_{name}"
            d = [rows[f][a] - rows[f][b] for f in rows]
            print(f"  ep vs {base:9s} {tag or 'K10man_'}{name}: gain {np.mean(d):5.2f} m, improved {sum(x > 0 for x in d)}/7, "
                  f"per-floor {np.round(d, 2).tolist()}")
print("done")
