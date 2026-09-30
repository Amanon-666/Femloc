"""后验预测检验（仅诊断）：LRM 零标注预测在留出上层楼层上的校准。严格留一楼层，与 lrm_v2 相同的先验拟合。"""
import numpy as np
from scipy.special import ndtr
from data.uji import load_floors
from models.radio_map import cells, fit_observation
from models.cross_floor import fit_transfer, predictive, ap_bin, nearest
from scripts.evaluate_scm_tobit import PAIRS
tr = load_floors("/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv")
floors = sorted({f for p in PAIRS for f in p})
C = {f: cells(tr[f]) for f in floors}
UB = np.array([-90.0, -80.0, -70.0, -60.0, -50.0])
det, cen, cal = [], [], []
for t, (lo, up) in enumerate(PAIRS):
    obs = fit_observation([C[f] for f in floors if f != up])
    tf = fit_transfer([(C[a], C[b]) for a, b in PAIRS if up not in (a, b)], obs)
    lc, uc = C[lo], C[up]
    mu, v, present, pos = predictive(lc, tf, obs)
    z, st = ap_bin(lc), obs["sigma_total"] ** 2
    j, d = nearest(uc["xy"], pos)
    U, J = np.flatnonzero(d <= 1.0), j[d <= 1.0]
    kU, nU, kL = uc["k"][U], uc["n"][U][:, None], lc["k"][J]
    mU, uL = uc["s1"][U] / np.maximum(kU, 1), lc["s1"][J] / np.maximum(kL, 1)
    M, V = mu[J], v[J]
    for store, sel in ((det, (kU >= 3) & (kL >= 3) & present[None]), (cen, (kU >= 3) & (kL == 0) & present[None])):
        i, a = np.nonzero(sel)
        r = mU[i, a] - M[i, a]
        store.append(np.stack([z[a], uL[i, a], r, r / np.sqrt(V[i, a] + st / kU[i, a]), t * 1000 + a], 1))
    p = np.clip(tf["h"] * ndtr((M - tf["theta"]) / np.sqrt(tf["s"] ** 2 + V)), 1e-9, 1)
    m = np.broadcast_to(present[None], p.shape)
    cal.append(np.stack([p[m], kU[m], np.broadcast_to(nU, p.shape)[m], (kL >= 1)[m]], 1))
D, Cn, K = np.concatenate(det), np.concatenate(cen), np.concatenate(cal)
print("A. 旧层检出单元 (k_L>=3, k_U>=3)：按 AP 档 z × 旧层检出水平 u")
print("   z  u-bin        n   mean(r)  rms(z)")
ub = np.digitize(D[:, 1], UB)
for zb in range(6):
    for b in range(len(UB) + 1):
        s = (D[:, 0] == zb) & (ub == b)
        if s.sum() >= 30:
            lo_ = "-inf" if b == 0 else int(UB[b - 1]); hi_ = "0" if b == len(UB) else int(UB[b])
            print(f"   {zb}  ({lo_},{hi_}] {s.sum():7d}  {D[s, 2].mean():7.2f}  {np.sqrt((D[s, 3] ** 2).mean()):6.2f}")
print("   within-AP slope of r on u (AP 组内去均值后回归), 按 z：")
for zb in range(6):
    s = D[:, 0] == zb
    g = D[s, 4]; _, gi = np.unique(g, return_inverse=True)
    cnt = np.bincount(gi); keep = cnt[gi] >= 4
    if keep.sum() < 50: continue
    u_ = D[s, 1][keep]; r_ = D[s, 2][keep]; gi = gi[keep]
    ud = u_ - (np.bincount(gi, u_) / np.maximum(np.bincount(gi), 1))[gi]
    rd = r_ - (np.bincount(gi, r_) / np.maximum(np.bincount(gi), 1))[gi]
    beta = (ud * rd).sum() / (ud * ud).sum()
    r2 = 1 - ((rd - beta * ud) ** 2).sum() / (rd ** 2).sum()
    print(f"     z={zb}: cells {keep.sum():6d}  slope {beta:6.3f}  within-AP var explained {r2:.3f}  sd(u within AP) {ud.std():.1f} dB")
print("B. 旧层删失、上层检出单元 (k_L=0, k_U>=3)：按 z")
for zb in range(6):
    s = Cn[:, 0] == zb
    if s.sum() >= 30:
        print(f"   z={zb}: n {s.sum():6d}  mean(r) {Cn[s, 2].mean():7.2f}  rms(z) {np.sqrt((Cn[s, 3] ** 2).mean()):5.2f}")
print("C. 检出概率校准（预测 p 十分位 → 实测检出率），分旧层检出/删失")
for lab, s in (("lower-detected", K[:, 3] == 1), ("lower-censored", K[:, 3] == 0)):
    q = np.quantile(K[s, 0], np.linspace(0, 1, 11)); b = np.clip(np.digitize(K[s, 0], q[1:-1]), 0, 9)
    out = [(round(K[s, 0][b == i].mean(), 3), round(K[s, 1][b == i].sum() / K[s, 2][b == i].sum(), 3)) for i in range(10) if (b == i).any()]
    print(f"   {lab}: {out}")
