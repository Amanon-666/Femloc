"""诊断（仅源楼层对，零标注）：跨楼层传播是否由三维路径损耗几何决定。
每个 AP 只用目标上层下方的楼层（同建筑）的可靠检出单元与删失单元拟合
  μ = P − 10n·log10(√(d² + (hΔf)²) + 1) − F·|Δf|，
AP 水平位置 q_a 与所在楼层 f_a 网格搜索，P 闭式，删失单元以 hinge 约束 μ < −90。
比较上层同坐标可靠检出单元的预测误差：平移模型 vs 物理均值 vs 物理 + ρ·旧层残差；再比较零标注定位。"""
import time
import numpy as np
from data.uji import load_floors
from models.radio_map import cells
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import mde, lower_map

R = "/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/"
tr, va = load_floors(R + "trainingData.csv"), load_floors(R + "validationData.csv")
TOP = {"B0": 3, "B1": 3, "B2": 4}
ALL = [f"{b}F{f}" for b, t in TOP.items() for f in range(t + 1)]
C = {f: cells(tr[f]) for f in ALL}
H, THC, STEP = 4.0, -90.0, 3.0
SET = [(n, F) for n in (2.0, 2.75, 3.5) for F in (6.0, 12.0, 18.0)]


def stats(c):
    q = c["k"] / c["n"][:, None]
    return (q >= 0.5) & (c["n"][:, None] >= 5), (c["k"] == 0) & (c["n"][:, None] >= 10), c["s1"] / np.maximum(c["k"], 1)


S = {f: stats(C[f]) for f in ALL}


def fit_aps(b, lower_floors):
    allxy = np.vstack([C[f"{b}F{f}"]["xy"] for f in range(TOP[b] + 1)])
    lo_, hi_ = allxy.min(0) - 15, allxy.max(0) + 15
    gx, gy = np.meshgrid(np.arange(lo_[0], hi_[0], STEP), np.arange(lo_[1], hi_[1], STEP))
    G = np.c_[gx.ravel(), gy.ravel()]
    out = {}
    for a in range(520):
        X, Fo, Y, Xc, Fc = [], [], [], [], []
        for f in lower_floors:
            rel, cen, m = S[f"{b}F{f}"]
            xy = C[f"{b}F{f}"]["xy"]
            i, j = np.flatnonzero(rel[:, a]), np.flatnonzero(cen[:, a])
            X.append(xy[i]); Fo.append(np.full(len(i), f)); Y.append(m[i, a])
            Xc.append(xy[j]); Fc.append(np.full(len(j), f))
        X, Fo, Y, Xc, Fc = np.vstack(X), np.concatenate(Fo), np.concatenate(Y), np.vstack(Xc), np.concatenate(Fc)
        if len(Y) < 5:
            continue
        rng = np.random.default_rng(a)
        if len(Y) > 300:
            s = rng.choice(len(Y), 300, replace=False); X, Fo, Y = X[s], Fo[s], Y[s]
        if len(Fc) > 150:
            s = rng.choice(len(Fc), 150, replace=False); Xc, Fc = Xc[s], Fc[s]
        d2 = ((G[:, None] - X[None]) ** 2).sum(-1)
        d2c = ((G[:, None] - Xc[None]) ** 2).sum(-1)
        best = {s: (np.inf, None) for s in SET}
        for fa in range(TOP[b] + 1):
            df, dfc = np.abs(Fo - fa), np.abs(Fc - fa)
            L = np.log10(np.sqrt(d2 + (H * df) ** 2) + 1)
            Lc = np.log10(np.sqrt(d2c + (H * dfc) ** 2) + 1)
            for n, F in SET:
                r = Y + 10 * n * L + F * df
                P = r.mean(1)
                sc = ((r - P[:, None]) ** 2).sum(1) + (np.maximum(P[:, None] - 10 * n * Lc - F * dfc - THC, 0) ** 2).sum(1)
                k = int(sc.argmin())
                if sc[k] < best[(n, F)][0]:
                    best[(n, F)] = (float(sc[k]), (G[k].copy(), int(fa), float(P[k])))
        out[a] = {s: v[1] for s, v in best.items()}
    return out


def predict(fits, s, P_xy, f):
    n, F = s
    mu = np.full((len(P_xy), 520), np.nan)
    for a, v in fits.items():
        q, fa, P = v[s]
        d = np.sqrt(((P_xy - q) ** 2).sum(-1))
        mu[:, a] = P - 10 * n * np.log10(np.sqrt(d ** 2 + (H * (f - fa)) ** 2) + 1) - F * abs(f - fa)
    return mu


pair_data = []
for lo, up in PAIRS:
    t0 = time.time()
    b, fL, fU = lo[:2], int(lo[-1]), int(up[-1])
    fits = fit_aps(b, range(fL + 1))
    PL = C[lo]["xy"]
    d = np.sqrt(((C[up]["xy"][:, None] - PL[None]) ** 2).sum(-1))
    iu = np.flatnonzero(d.min(1) <= 1.0); jl = d.argmin(1)[iu]
    pd = {"lo": lo, "up": up, "fits": fits, "iu": iu, "jl": jl, "fL": fL, "fU": fU,
          "muU": {s: predict(fits, s, PL, fU) for s in SET}, "muL": {s: predict(fits, s, PL, fL) for s in SET}}
    pair_data.append(pd)
    print(f"{lo}->{up}: fitted APs {len(fits)}  {time.time() - t0:.0f}s", flush=True)

# 平移常数（诊断用：所有源对两层都可靠检出单元的中位差）
diffs = []
for pd in pair_data:
    relL, _, mL = S[pd["lo"]]; relU, _, mU = S[pd["up"]]
    both = relL[pd["jl"]] & relU[pd["iu"]]
    diffs.append((mU[pd["iu"]] - mL[pd["jl"]])[both])
bshift = float(np.median(np.concatenate(diffs)))
print("shift constant (median upper−lower on co-located reliable cells):", round(bshift, 2))

print("setting      rho   RMSE_both: shift  phys  phys+rho | RMSE_Lcensored: phys  (n)")
summ = {}
for s in SET:
    num, den = 0.0, 0.0
    for pd in pair_data:
        relL, _, mL = S[pd["lo"]]; relU, _, mU = S[pd["up"]]
        muU, muL = pd["muU"][s][pd["jl"]], pd["muL"][s][pd["jl"]]
        both = relL[pd["jl"]] & relU[pd["iu"]] & np.isfinite(muU)
        rL, rU = (mL[pd["jl"]] - muL)[both], (mU[pd["iu"]] - muU)[both]
        num += (rL * rU).sum(); den += (rL * rL).sum()
    rho = num / den
    e = {"shift": [], "phys": [], "physrho": [], "cens": []}
    for pd in pair_data:
        relL, cenL, mL = S[pd["lo"]]; relU, _, mU = S[pd["up"]]
        muU, muL = pd["muU"][s][pd["jl"]], pd["muL"][s][pd["jl"]]
        yU = mU[pd["iu"]]
        both = relL[pd["jl"]] & relU[pd["iu"]] & np.isfinite(muU)
        e["shift"].append((yU - mL[pd["jl"]] - bshift)[both])
        e["phys"].append((yU - muU)[both])
        e["physrho"].append((yU - muU - rho * (mL[pd["jl"]] - muL))[both])
        lc = ~relL[pd["jl"]] & (mL[pd["jl"]] * 0 == 0) & (S[pd["lo"]][1][pd["jl"]]) & relU[pd["iu"]] & np.isfinite(muU)
        e["cens"].append((yU - muU)[lc])
    r = {k: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for k, v in e.items()}
    r["n_cens"] = int(sum(len(x) for x in e["cens"])); r["rho"] = float(rho)
    summ[s] = r
    print(f"n={s[0]:<4} F={s[1]:<4} {rho:5.2f}   {r['shift']:6.2f} {r['phys']:6.2f} {r['physrho']:7.2f}   |   {r['cens']:6.2f}  ({r['n_cens']})")
best = min(SET, key=lambda s: summ[s]["physrho"])
print("best setting by physrho RMSE:", best)

# 按 AP 拟合楼层分类的误差（最佳设置）
cls = {"below(<fL)": [], "on fL": [], "on fU": [], "above(>fU)": []}
for pd in pair_data:
    relL, _, mL = S[pd["lo"]]; relU, _, mU = S[pd["up"]]
    muU, muL = pd["muU"][best][pd["jl"]], pd["muL"][best][pd["jl"]]
    yU = mU[pd["iu"]]
    both = relL[pd["jl"]] & relU[pd["iu"]] & np.isfinite(muU)
    rho = summ[best]["rho"]
    for a, v in pd["fits"].items():
        fa = v[best][1]
        key = "below(<fL)" if fa < pd["fL"] else "on fL" if fa == pd["fL"] else "on fU" if fa == pd["fU"] else "above(>fU)"
        sel = both[:, a]
        if sel.any():
            cls[key].append(np.stack([(yU[:, a] - mL[pd["jl"]][:, a] - bshift)[sel],
                                      (yU[:, a] - muU[:, a] - rho * (mL[pd["jl"]][:, a] - muL[:, a]))[sel]], 1))
for k, v in cls.items():
    if v:
        v = np.concatenate(v)
        print(f"  class {k:11s} cells {len(v):6d}  RMSE shift {np.sqrt((v[:, 0] ** 2).mean()):6.2f}  phys+rho {np.sqrt((v[:, 1] ** 2).mean()):6.2f}")

# 零标注定位（Tobit σ=16, θ=−80），候选 = 旧层坐标
print("zero-shot localization (Tobit 16/−80, candidates = lower-floor positions): val | train-all(≤1500)")
agg = {"old": [], "shift": [], "phys": []}
for pd in pair_data:
    lo, up = pd["lo"], pd["up"]
    relL, cenL, mL = S[lo]
    old, PL = lower_map(tr[lo])
    muU, muL = pd["muU"][best], pd["muL"][best]
    rho = summ[best]["rho"]
    phys = np.where(np.isfinite(muU), muU + np.where(relL, rho * (mL - np.nan_to_num(muL)), 0.0), np.where(relL, mL + bshift, -110.0))
    shift = np.where(old > -109.9, np.clip(old + bshift, -110, 0), -110.0)
    maps = {"old": old, "shift": shift, "phys": np.clip(phys, -110, -20)}
    rng = np.random.default_rng(0)
    it = rng.choice(len(tr[up]["xy"]), min(1500, len(tr[up]["xy"])), replace=False)
    row = []
    for k, M in maps.items():
        ev = mde(tobit_match(va[up]["rssi"], M, PL, 16.0, -80.0), va[up]["xy"])
        et = mde(tobit_match(tr[up]["rssi"][it], M, PL, 16.0, -80.0), tr[up]["xy"][it])
        agg[k].append((ev, et)); row.append(f"{k} {ev:6.2f} | {et:6.2f}")
    print(f"  {up}: " + "   ".join(row), flush=True)
for k, v in agg.items():
    v = np.array(v); print(f"  EQUAL {k:6s} val {v[:, 0].mean():6.2f}  train {v[:, 1].mean():6.2f}")
print("reference: LRM zero-shot (m2) val 14.10 int 11.82; old-map Tobit val 17.34")
print("done")
