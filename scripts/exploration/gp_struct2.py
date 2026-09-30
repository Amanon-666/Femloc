"""GP 边缘似然比较跨楼层残差的协方差结构（仅源楼层对；上下层都可靠检出的共位单元，q≥0.5、n≥5）。
y_U = CEN + ρ_z(y_L − CEN) + β_z + δ，同一 AP 的单元之间 δ 的协方差：
  S0：ρ≡1，s0_z²·11ᵀ + s2²·RBF(ℓ)，ℓ ∈ [1, 300]      （每 AP 常数 + 单一尺度）
  S1：ρ_z 自由，其余同 S0
  S3：ρ≡1，s0_z²·11ᵀ + s2²·RBF(ℓ2∈[1,20]) + s1_z²·RBF(ℓ1∈[20,300])   （短 + 长两尺度）
  S2：ρ_z 自由 + 两尺度
对角 η² + σ_t²/k_U + ρ²σ_t²/k_L。参数有界（sigmoid），比较留一楼层的留出负对数似然（每单元 nats）。
要回答的问题：跨楼层残差里是否有长程（楼层尺度）成分。若有，潜在 EP 用的单一 12 m 尺度就是结构性错设。"""
import time
import numpy as np
import torch
from data.uji import load_floors
from models.radio_map import cells, fit_observation
from models.cross_floor import ap_bin, nearest, NB
from scripts.evaluate_scm_tobit import PAIRS
torch.set_default_dtype(torch.float64); torch.set_num_threads(8)
tr = load_floors("/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv")
floors = sorted({f for p in PAIRS for f in p})
C = {f: cells(tr[f]) for f in floors}
st = fit_observation([C[f] for f in floors])["sigma_total"] ** 2
CEN = -75.0
S = ("S0", "S1", "S3", "S2")
TWO = ("S2", "S3")
BND = {"rho": (0.0, 2.0), "ls0": (0.05, 30.0), "ls1": (0.05, 30.0), "ls2": (0.05, 30.0), "leta": (0.3, 20.0),
       "ll1": (20.0, 300.0), "ll2_two": (1.0, 20.0), "ll2_one": (1.0, 300.0)}


def groups(pairs):
    G = []
    for lo, up in pairs:
        lc, uc = C[lo], C[up]; z = ap_bin(lc)
        l, d = nearest(uc["xy"], lc["xy"]); u = np.flatnonzero(d <= 1.0); l = l[u]
        kL, kU = lc["k"][l], uc["k"][u]; nL, nU = lc["n"][l][:, None], uc["n"][u][:, None]
        ok = (kL / nL >= .5) & (kU / nU >= .5) & (nL >= 5) & (nU >= 5)
        mL, mU = lc["s1"][l] / np.maximum(kL, 1), uc["s1"][u] / np.maximum(kU, 1)
        for a in np.flatnonzero(ok.sum(0) >= 3):
            i = np.flatnonzero(ok[:, a])
            G.append(dict(y=mU[i, a], u=mL[i, a], xy=uc["xy"][u][i], kU=kU[i, a], kL=kL[i, a], z=int(z[a])))
    return G


def pack(G):
    M, n = max(len(g["y"]) for g in G), len(G)
    Y, U, KU, KL, MK = (np.zeros((n, M)) for _ in range(5)); XY = np.zeros((n, M, 2)); Z = np.zeros(n, int)
    for i, g in enumerate(G):
        m = len(g["y"])
        Y[i, :m], U[i, :m], KU[i, :m], KL[i, :m], MK[i, :m], XY[i, :m], Z[i] = g["y"], g["u"], g["kU"], g["kL"], 1, g["xy"], g["z"]
    KU[MK == 0] = 1; KL[MK == 0] = 1
    T = torch.tensor
    return dict(Y=T(Y), U=T(U), KU=T(KU), KL=T(KL), MK=T(MK), D2=T(((XY[:, :, None] - XY[:, None]) ** 2).sum(-1)), Z=T(Z))


def val(P, k, s):
    if k == "beta":
        return P[k]
    lo, hi = BND["ll2_two" if (k == "ll2" and s in TWO) else "ll2_one" if k == "ll2" else k]
    return lo + (hi - lo) * torch.sigmoid(P[k])


def raw(k, v, s, n=None):
    lo, hi = BND["ll2_two" if (k == "ll2" and s in TWO) else "ll2_one" if k == "ll2" else k]
    r = float(np.log((v - lo) / (hi - v)))
    return torch.full((n,), r) if n else torch.tensor(r)


def nll(P, B, s):
    Z, MK = B["Z"], B["MK"]; mm = MK[:, :, None] * MK[:, None]
    rho = val(P, "rho", s)[Z] if s in ("S1", "S2") else torch.ones(len(Z))
    K = val(P, "ls0", s)[Z][:, None, None] ** 2 + val(P, "ls2", s) ** 2 * torch.exp(-B["D2"] / (2 * val(P, "ll2", s) ** 2))
    if s in TWO:
        K = K + val(P, "ls1", s)[Z][:, None, None] ** 2 * torch.exp(-B["D2"] / (2 * val(P, "ll1", s) ** 2))
    dg = val(P, "leta", s) ** 2 + st / B["KU"] + rho[:, None] ** 2 * st / B["KL"]
    K = K * mm + torch.diag_embed(dg * MK + (1 - MK) + 1e-6)
    r = (B["Y"] - CEN - P["beta"][Z][:, None] - rho[:, None] * (B["U"] - CEN)) * MK
    L = torch.linalg.cholesky(K)
    a = torch.cholesky_solve(r[..., None], L)[..., 0]
    return 0.5 * (r * a).sum(1) + torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1) + 0.5 * MK.sum(1) * np.log(2 * np.pi)


def fit(B, s):
    P = {"rho": raw("rho", 1.0, s, NB), "beta": torch.zeros(NB), "ls0": raw("ls0", 5.0, s, NB), "ls1": raw("ls1", 4.0, s, NB),
         "ll1": raw("ll1", 60.0, s), "ls2": raw("ls2", 4.0, s), "ll2": raw("ll2", 8.0, s), "leta": raw("leta", 3.0, s)}
    P = {k: v.clone().requires_grad_(True) for k, v in P.items()}
    n = B["MK"].sum()
    opt = torch.optim.LBFGS(list(P.values()), max_iter=400, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad(); f = nll(P, B, s).sum() / n; f.backward(); return f
    opt.step(closure)
    P = {k: v.detach() for k, v in P.items()}
    with torch.no_grad():
        return P, float(nll(P, B, s).sum() / n)


def show(P, s):
    e = lambda k: np.round(val(P, k, s).numpy(), 2)
    txt = f"rho {e('rho') if s in ('S1', 'S2') else 1}  beta {np.round(P['beta'].numpy(), 1)}  s0 {e('ls0')}"
    txt += f"  s_short {float(val(P, 'ls2', s)):.2f} l_short {float(val(P, 'll2', s)):.1f}"
    if s in TWO:
        txt += f"  s_long {e('ls1')} l_long {float(val(P, 'll1', s)):.1f}"
    return txt + f"  eta {float(val(P, 'leta', s)):.2f}"


t0 = time.time()
B = pack(groups(PAIRS))
print(f"groups {len(B['Z'])}  cells {int(B['MK'].sum())}  per-bin {np.bincount(B['Z'].numpy(), minlength=NB).tolist()}", flush=True)
for s in S:
    P, v = fit(B, s)
    print(f"{s}: train nll/cell {v:.4f}  {show(P, s)}", flush=True)
print(f"full fits {time.time() - t0:.0f}s", flush=True)
hold = {s: [] for s in S}
for lo, up in PAIRS:
    Btr, Bte = pack(groups([(a, b) for a, b in PAIRS if up not in (a, b)])), pack(groups([(lo, up)]))
    row = []
    for s in S:
        P, _ = fit(Btr, s)
        with torch.no_grad():
            h = float(nll(P, Bte, s).sum() / Bte["MK"].sum())
        hold[s].append(h); row.append(f"{s} {h:.4f}")
    print(f"  held {up} cells {int(Bte['MK'].sum()):5d}: " + "  ".join(row), flush=True)
print("LOPO mean held-out nll/cell: " + "  ".join(f"{s} {np.mean(v):.4f}" for s, v in hold.items()))
for a, b in (("S3", "S0"), ("S1", "S0"), ("S2", "S1"), ("S2", "S0")):
    d = np.array(hold[a]) - np.array(hold[b])
    print(f"  {a} − {b}: mean {d.mean():+.4f}, better in {(d < 0).sum()}/7 folds, per fold {np.round(d, 4).tolist()}")
print(f"done {time.time() - t0:.0f}s")
