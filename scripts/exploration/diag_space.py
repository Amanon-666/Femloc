"""诊断：跨楼层残差应放在哪个空间（只用 7 对源转移 trainingData，去重观测）。"""
import numpy as np
from data.uji import load_floors
PAIRS = [(f"B{b}F{f}", f"B{b}F{f + 1}") for b, top in ((0, 3), (1, 3), (2, 4)) for f in range(top - 1)]
FLOORS = sorted({f for p in PAIRS for f in p})
tr = load_floors("/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv")

def cells(fl):
    _, first = np.unique(fl["groups"], return_index=True)
    r, xy = fl["rssi"][first], fl["xy"][first]
    keys, inv = np.unique(xy, axis=0, return_inverse=True); inv = inv.ravel()
    det = r != 100; x = np.where(det, r, 0).astype(float); n = np.bincount(inv).astype(float)
    k, s, s2 = (np.zeros((len(keys), 520)) for _ in range(3))
    np.add.at(k, inv, det); np.add.at(s, inv, x); np.add.at(s2, inv, x * x)
    m = np.where(k > 0, s / np.maximum(k, 1), np.nan)
    sd = np.where(k > 1, np.sqrt(np.maximum(s2 - k * np.nan_to_num(m) ** 2, 0) / np.maximum(k - 1, 1)), np.nan)
    return keys, n, k, k / n[:, None], m, sd, (s - 110.0 * (n[:, None] - k)) / n[:, None]

def loo_r2(v, g, mn=3):
    v = np.asarray(v, float); _, gi, c = np.unique(np.asarray(g), return_inverse=True, return_counts=True)
    keep = c[gi] >= mn; v, gi = v[keep], gi[keep]
    tot, cnt = np.bincount(gi, v), np.bincount(gi); pred = (tot[gi] - v) / (cnt[gi] - 1)
    return 1 - ((v - pred) ** 2).sum() / ((v - v.mean()) ** 2).sum(), len(v), len(np.unique(gi))

pool = {"latent(both det)": ([], []), "imputed(same cells)": ([], []), "imputed(all relevant)": ([], [])}
offs = []
print("pair            pos  cells  global   R2_lat  R2_imp_same  R2_imp_rel")
for lo, up in PAIRS:
    kL, nL, cL, qL, mL, _, iL = cells(tr[lo]); kU, nU, cU, qU, mU, _, iU = cells(tr[up])
    d = np.sqrt(((kU[:, None] - kL[None]) ** 2).sum(-1)); U = np.flatnonzero(d.min(1) <= 1.0); L = d.argmin(1)[U]
    enough = (nU[U][:, None] >= 5) & (nL[L][:, None] >= 5)
    p, a = np.nonzero(enough & (qU[U] >= 0.5) & (qL[L] >= 0.5))
    pr, ar = np.nonzero(enough & ((qU[U] > 0) | (qL[L] > 0)))
    lat, ims, imr = mU[U][p, a] - mL[L][p, a], iU[U][p, a] - iL[L][p, a], iU[U][pr, ar] - iL[L][pr, ar]
    tag = f"{lo}->{up}"
    print(f"{tag:14s} {len(U):4d} {len(lat):6d} {lat.mean():7.2f}  {loo_r2(lat, a)[0]:7.3f}  {loo_r2(ims, a)[0]:11.3f}  {loo_r2(imr, ar)[0]:10.3f}")
    for key, v, g in (("latent(both det)", lat, a), ("imputed(same cells)", ims, a), ("imputed(all relevant)", imr, ar)):
        pool[key][0].append(v); pool[key][1].append(np.char.add(tag + ":", g.astype(str)))
    maxL = np.nanmax(np.where(qL >= 0.5, mL, np.nan), axis=0)
    for ap in np.unique(a):
        sel = a == ap
        if sel.sum() >= 3:
            offs.append((lat[sel].mean(), lat[sel].std(ddof=1), maxL[ap]))
for key, (v, g) in pool.items():
    r2, n, nap = loo_r2(np.concatenate(v), np.concatenate(g)); print(f"POOLED {key:22s} LOO-R2 per-AP const = {r2:.3f}  (cells {n}, APs {nap})")
o = np.array(offs)
print("per-AP offsets (latent): n", len(o), "quantiles 5/25/50/75/95:", np.round(np.percentile(o[:, 0], [5, 25, 50, 75, 95]), 1),
      "| >+3:", round((o[:, 0] > 3).mean(), 3), "<-3:", round((o[:, 0] < -3).mean(), 3), "| median within-AP sd", round(np.median(o[:, 1]), 2))
print("histogram (5 dB bins -30..30):", np.histogram(np.clip(o[:, 0], -30, 30), bins=np.arange(-30, 31, 5))[0].tolist())
for lo_, hi_ in ((-110, -80), (-80, -70), (-70, -60), (-60, -50), (-50, 0)):
    s = (o[:, 2] > lo_) & (o[:, 2] <= hi_)
    if s.sum(): print(f"  lower-floor max RSS ({lo_},{hi_}]: APs {s.sum():4d}  mean offset {o[s, 0].mean():6.2f}  frac>0 {(o[s, 0] > 0).mean():.2f}")
M, Q, SD, X, N = [], [], [], [], []
for f in FLOORS:
    _, n, k, q, m, sd, _ = cells(tr[f]); ok = (n[:, None] >= 8) & (k >= 1)
    M.append(m[ok]); Q.append(q[ok]); SD.append(sd[(k >= 5) & (q >= 0.9)]); N.append(n)
    r = tr[f]["rssi"]; X.append(r[r != 100])
M, Q, SD, X, N = map(np.concatenate, (M, Q, SD, X, N))
print("detected RSS quantiles 0/0.1/1/5%:", np.percentile(X, [0, 0.1, 1, 5]).tolist(), "| scans/position median", np.median(N))
print("scan sd (cells q>=0.9, k>=5): median", round(np.median(SD), 2), "mean", round(SD.mean(), 2), "n", len(SD))
for lo_, hi_ in ((-105, -100), (-100, -95), (-95, -90), (-90, -85), (-85, -80), (-80, -75), (-75, -70), (-70, -60), (-60, -50), (-50, 0)):
    s = (M > lo_) & (M <= hi_); print(f"  mean-detected ({lo_},{hi_}]: cells {s.sum():6d}  miss rate {1 - Q[s].mean():.3f}")
