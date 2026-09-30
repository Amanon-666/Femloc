"""跨楼层潜在无线电地图（LRM）：同一个观测模型贯穿先验拟合、锚点更新和匹配。

目标层 AP a 在位置 p 的潜在水平 ℓ_a(p)（dBm）。每条扫描：检出 ~ Bernoulli(h·Φ((ℓ−θ)/s))；
检出时 x = ℓ + c + ε，ε ~ N(0, σ²)，c ~ N(0, τ²) 为整条扫描的共同偏移。
跨楼层先验 ℓ_a(p) = CEN + ρ_z(ℓ^L_a(p) − CEN) + b_a + g_a(p) + e_a(p)：
  ρ_z：Kennedy–O'Hagan 自回归系数，即迁移 GP 的域相关系数，按 AP 档由删失复合似然估计；ρ≡1 退化为平移；
  ℓ^L：旧层潜在水平。旧层检出的单元取检出均值（测量方差 σ_t²/k_L 并入 e）；未检出单元是删失，
       目标层潜在水平的先验为 N(μ0(z_a), s_b² + s_c² + nug + v_L0(z_a))；
  b_a ~ N(b̄(z_a), s_b²(z_a))：穿过楼板的整体偏移，z_a 为 AP 在旧层的最强可靠检出档（代理 AP 所在楼层）；
  g_a ~ GP(0, s_c²·RBF_ℓ)：空间相关偏差；e ~ N(0, nug)：单元独立偏差。
锚点每个位置的检出次数和检出值都按同一观测模型进入后验；未检出是删失证据，不丢弃、也不当作 −110。
非高斯后验用 EP（Gauss–Hermite 矩匹配）近似，候选边缘 N(μ_p, v_p) 再代回同一观测模型匹配。
先验参数只由源楼层对估计：方差分量用矩估计，偏移均值、检出链接和删失单元先验用复合似然。
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtr

EDGES = np.array([-80.0, -70.0, -60.0, -50.0])
NB = len(EDGES) + 2
DBINS = np.array([0, 2, 4, 7, 10, 14, 19, 25, 32, 40, 50, 65, 80])
ELLS = (3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 60.0)
GH_T, GH_W = np.polynomial.hermite.hermgauss(32)
LOG_GH_W = np.log(GH_W / np.sqrt(np.pi))


def ap_bin(lc):
    q = lc["k"] / lc["n"][:, None]
    m = np.where(q >= 0.5, lc["s1"] / np.maximum(lc["k"], 1), -np.inf).max(0)
    return np.where(np.isfinite(m), 1 + np.searchsorted(EDGES, m), 0)


def nearest(xy, ref):
    d = np.sqrt(((xy[:, None] - ref[None]) ** 2).sum(-1))
    return d.argmin(1), d.min(1)


def tilted(mu, v, n, k, m, st, h, theta, s):
    """N(ℓ; μ, v)·Binom(k; n, h·Φ((ℓ−θ)/s))·Π_j N(x_j; ℓ, σ_t²)：返回对数归一化常数（略去与参数无关的项）、
    均值和方差。检出值项对 ℓ 是高斯，先解析合并；检出次数项用 Gauss–Hermite 积分。"""
    kk = np.maximum(k, 1)
    has = k > 0
    vc = np.where(has, 1 / (1 / v + kk / st), v)
    mc = np.where(has, vc * (mu / v + kk * m / st), mu)
    val = np.where(has, -0.5 * (m - mu) ** 2 / (v + st / kk) - 0.5 * np.log(v + st / kk), 0.0)
    L = mc[..., None] + np.sqrt(2 * vc)[..., None] * GH_T
    g = np.clip(h * ndtr((L - theta) / s), 1e-12, 1 - 1e-12)
    lp = k[..., None] * np.log(g) + (n - k)[..., None] * np.log1p(-g) + LOG_GH_W
    lz = logsumexp(lp, axis=-1, keepdims=True)
    w = np.exp(lp - lz)
    mean = (w * L).sum(-1)
    var = np.maximum((w * (L - mean[..., None]) ** 2).sum(-1), 1e-9)
    return val + lz[..., 0], mean, var


CEN = -75.0


def _pair_arrays(pair_cells):
    """上下层共位单元：BOTH 两层都检出（矩估计）、LINK 旧层检出（复合似然）、UND 旧层删失。"""
    BOTH, G, XY, LINK = [], [], [], []
    UND = [[] for _ in range(NB)]
    for t, (lc, uc) in enumerate(pair_cells):
        z = ap_bin(lc)
        l, dist = nearest(uc["xy"], lc["xy"])
        u = np.flatnonzero(dist <= 1.0)
        l = l[u]
        kL, kU = lc["k"][l], uc["k"][u]
        nU = np.repeat(uc["n"][u][:, None], kL.shape[1], 1)
        mL, mU = lc["s1"][l] / np.maximum(kL, 1), uc["s1"][u] / np.maximum(kU, 1)
        uni = (lc["k"].sum(0) > 0) | (kU.sum(0) > 0)
        zz = np.broadcast_to(z, kL.shape)
        det = (kL >= 1) & uni
        i, a = np.nonzero(det & (kU >= 1))
        BOTH.append(np.stack([mU[i, a], mL[i, a], kU[i, a], kL[i, a], z[a]], 1))
        G.append(a + 1000 * t)
        XY.append(uc["xy"][u][i])
        LINK.append(np.stack([mL[det], kL[det], zz[det], kU[det], nU[det], mU[det]], 1))
        for b in range(NB):
            sel = (kL == 0) & uni & (zz == b)
            UND[b].append(np.stack([kU[sel], nU[sel], mU[sel]], 1))
    return np.concatenate(BOTH), np.concatenate(G), np.concatenate(XY), np.concatenate(LINK), UND


def _link_terms(tf, obs, link):
    lm, lk, lz, lku, lnu, lmu = link.T
    lz = lz.astype(int)
    st, r = obs["sigma_total"] ** 2, tf["rho"][lz]
    mu = CEN + r * (lm - CEN) + tf["bbar"][lz]
    v = tf["sb2"][lz] + tf["s_w2"] + r ** 2 * st / lk
    return tilted(mu, v, lnu, lku, lmu, st, tf["h"], tf["theta"], tf["s"])[0]


def link_nll(pair_cells, tf, obs):
    """给定楼层对上、旧层检出单元的删失复合负对数似然（每单元）；留出楼层对上用于结构比较。"""
    return float(-_link_terms(tf, obs, _pair_arrays(pair_cells)[3]).mean())


def fit_transfer(pair_cells, obs, ell=None, slope=True, sweeps=3):
    """源楼层对上估计跨楼层先验。旧层检出单元的先验均值 ℓ = CEN + ρ_z(m_L − CEN) + β_z：
    Kennedy–O'Hagan 自回归迁移，ρ_z 即迁移 GP 的域相关系数（slope=False 时 ρ≡1，退化为平移）。
    β、ρ 与检出链接 (θ, s) 用删失复合似然；方差分量用 ρ 调整后残差的矩估计；两者交替 sweeps 轮。"""
    st, h = obs["sigma_total"] ** 2, obs["h_det"]
    BOTH, G, XY, LINK, UND = _pair_arrays(pair_cells)
    _, gi, gn = np.unique(G, return_inverse=True, return_counts=True)
    Zb = BOTH[:, 4].astype(int)
    gz = np.round(np.bincount(gi, Zb) / gn).astype(int)
    big, ok = gn >= 3, gn[gi] >= 3

    def components(rho):
        r = rho[Zb]
        R = BOTH[:, 0] - CEN - r * (BOTH[:, 1] - CEN)
        N = st * (1 / BOTH[:, 2] + r ** 2 / BOTH[:, 3])
        bhat = np.bincount(gi, R) / gn
        e = R - bhat[gi]
        s_w2 = max(float(np.mean(e[ok] ** 2 * gn[gi][ok] / (gn[gi][ok] - 1) - N[ok])), 1.0)
        gnoise = np.bincount(gi, N) / gn
        b0, sb2 = np.zeros(NB), np.zeros(NB)
        for b in range(NB):
            sel = big & (gz == b)
            sel = sel if sel.sum() >= 5 else big
            b0[b] = bhat[sel].mean()
            sb2[b] = max(bhat[sel].var() - np.mean((s_w2 + gnoise[sel]) / gn[sel]), 1.0)
        return e, s_w2, b0, sb2

    k_ = NB if slope else 0
    rho, x = np.ones(NB), None
    for _ in range(sweeps if slope else 1):
        e, s_w2, b0, sb2 = components(rho)
        cur = {"sb2": sb2, "s_w2": s_w2, "h": h}

        def nll(p):
            cur.update(bbar=p[:NB], rho=p[NB:NB + k_] if slope else np.ones(NB),
                       theta=p[NB + k_], s=np.exp(p[NB + k_ + 1]))
            return -_link_terms(cur, obs, LINK).mean()

        start = np.r_[b0, rho[:k_], -85.0, np.log(12.0)] if x is None else x
        bounds = [(None, None)] * NB + [(0.0, 1.5)] * k_ + [(None, None), (np.log(0.5), np.log(60.0))]
        x = minimize(nll, start, method="L-BFGS-B", bounds=bounds).x
        if slope:
            rho = x[NB:NB + k_].copy()
    bbar, theta, s = x[:NB], float(x[NB + k_]), float(np.exp(x[NB + k_ + 1]))
    link = float(nll(x))
    # 空间相关：ρ 调整后、按 AP 去均值的残差在距离分箱上的平均乘积 / s_w²，拟合 frac·RBF_ℓ（1−frac 为 nugget）
    num, den = np.zeros(len(DBINS) - 1), np.zeros(len(DBINS) - 1)
    order, starts = np.argsort(gi, kind="stable"), np.concatenate([[0], np.cumsum(gn)])
    for g in np.flatnonzero(gn >= 4):
        idx = order[starts[g]:starts[g + 1]]
        iu = np.triu_indices(len(idx), 1)
        d = np.sqrt(((XY[idx][:, None] - XY[idx][None]) ** 2).sum(-1))[iu]
        bi = np.digitize(d, DBINS) - 1
        keep = (bi >= 0) & (bi < len(num))
        np.add.at(num, bi[keep], np.outer(e[idx], e[idx])[iu][keep])
        np.add.at(den, bi[keep], 1)
    corr, mid = num / np.maximum(den, 1) / s_w2, 0.5 * (DBINS[:-1] + DBINS[1:])
    fits = []
    for length in (ELLS if ell is None else (ell,)):
        kk = np.exp(-mid ** 2 / (2 * length * length))
        frac = float(np.clip((den * kk * corr).sum() / (den * kk * kk).sum(), 0, 1))
        fits.append(((den * (corr - frac * kk) ** 2).sum(), length, frac))
    _, ell, frac = min(fits)
    sc2, nug = frac * s_w2, (1 - frac) * s_w2
    # 旧层删失单元：每档一个先验 N(μ0, s_b²+s_c²+nug+v_L0)，同一似然；k=0 单元按 n 聚合
    mu0, vL0 = np.zeros(NB), np.zeros(NB)
    pooled = np.concatenate([np.concatenate(x_) for x_ in UND])
    for b in range(NB):
        arr = np.concatenate(UND[b])
        arr = arr if arr[:, 1].sum() >= 200 else pooled
        hit = arr[arr[:, 0] >= 1]
        n0, c0 = np.unique(arr[arr[:, 0] == 0, 1], return_counts=True)
        kk, nn = np.r_[hit[:, 0], np.zeros(len(n0))], np.r_[hit[:, 1], n0]
        mm, ww = np.r_[hit[:, 2], np.zeros(len(n0))], np.r_[np.ones(len(hit)), c0]
        one, shared = np.ones(len(kk)), sb2[b] + sc2 + nug
        nll_c = lambda y: -(ww * tilted(y[0] * one, (shared + np.exp(y[1])) * one, nn, kk, mm, st, h, theta, s)[0]).sum() / ww.sum()
        y = minimize(nll_c, [-100.0, np.log(100.0)], method="L-BFGS-B").x
        mu0[b], vL0[b] = y[0], float(np.exp(y[1]))
    return {"bbar": bbar, "rho": rho, "sb2": sb2, "s_w2": s_w2, "sc2": sc2, "nug": nug, "ell": ell,
            "h": h, "theta": theta, "s": s, "mu0": mu0, "vL0": vL0, "corr": corr, "corr_d": mid,
            "link_nll": link, "n_cells": int(len(BOTH)), "n_link_cells": int(len(LINK)), "n_groups": int(big.sum())}


def predictive(lc, tf, obs, ac=None, censored=True, sweeps=20, damping=0.5):
    """候选潜在水平的边缘 N(μ_p, v_p)。ac 为锚点单元；censored=False 时只用检出过该 AP 的锚点（消融）。"""
    st, h, theta, s = obs["sigma_total"] ** 2, tf["h"], tf["theta"], tf["s"]
    z, pos = ap_bin(lc), lc["xy"]
    if ac is not None:
        _, d = nearest(ac["xy"], pos)
        pos = np.vstack([pos, ac["xy"][d > 0.5]])
    base, _ = nearest(pos, lc["xy"])
    kL = lc["k"][base]
    det = kL >= 1
    zb = np.broadcast_to(z, kL.shape)
    rho = tf["rho"][zb]
    mu = np.where(det, CEN + rho * (lc["s1"][base] / np.maximum(kL, 1) - CEN) + tf["bbar"][zb], tf["mu0"][zb])
    e = tf["nug"] + np.where(det, rho ** 2 * st / np.maximum(kL, 1), tf["vL0"][zb])
    sb2 = tf["sb2"][z]
    v = sb2[None] + tf["sc2"] + e
    present = lc["k"].sum(0) > 0
    if ac is not None:
        present = present | (ac["k"].sum(0) > 0)
        aps = np.flatnonzero(present)
        aj, _ = nearest(ac["xy"], pos)
        rbf = lambda p1, p2: np.exp(-((p1[:, None] - p2[None]) ** 2).sum(-1) / (2 * tf["ell"] ** 2))
        same_a = (aj[:, None] == aj[None]).astype(float)
        same_p = (np.arange(len(pos))[:, None] == aj[None]).astype(float)
        eA, eP = e[aj][:, aps].T, e[:, aps].T
        SA = sb2[aps][:, None, None] + tf["sc2"] * rbf(pos[aj], pos[aj])[None] + same_a[None] * eA[:, :, None]
        SPA = sb2[aps][:, None, None] + tf["sc2"] * rbf(pos, pos[aj])[None] + same_p[None] * eP[:, :, None]
        muA = mu[aj][:, aps].T
        kA = ac["k"][:, aps].T
        nA = np.broadcast_to(ac["n"][None], kA.shape)
        mA = (ac["s1"][:, aps] / np.maximum(ac["k"][:, aps], 1)).T
        use = np.ones(kA.shape, bool) if censored else kA >= 1
        ts, ns = np.zeros_like(muA), np.zeros_like(muA)
        eye, dA = np.eye(len(aj))[None], np.diagonal(SA, axis1=1, axis2=2)
        for _ in range(sweeps):
            sol = np.linalg.solve(eye + ts[:, :, None] * SA,
                                  np.concatenate([(ns - ts * muA)[:, :, None], ts[:, :, None] * SA], 2))
            pm = muA + np.einsum("akj,aj->ak", SA, sol[..., 0])
            pv = dA - np.einsum("akj,ajk->ak", SA, sol[..., 1:])
            tc, nc = 1 / pv - ts, pm / pv - ns
            ok = use & (tc > 1e-10)
            vc = 1 / np.where(ok, tc, 1.0)
            mc = nc * vc
            _, mh, vh = tilted(mc, vc, nA, kA, mA, st, h, theta, s)
            tn = 1 / vh - tc
            nn = np.where(tn > 0, mh / vh - nc, (mh - mc) * tc)
            ts = np.where(ok, (1 - damping) * ts + damping * np.maximum(tn, 0.0), ts)
            ns = np.where(ok, (1 - damping) * ns + damping * nn, ns)
        sol = np.linalg.solve(eye + ts[:, :, None] * SA,
                              np.concatenate([(ns - ts * muA)[:, :, None], ts[:, :, None] * SPA.transpose(0, 2, 1)], 2))
        mu[:, aps] += np.einsum("apk,ak->pa", SPA, sol[..., 0])
        v[:, aps] -= np.einsum("apk,akp->pa", SPA, sol[..., 1:])
    return mu, v, present, pos


def latent_loglik(q_rssi, mu, vlat, present, tf, obs, exact=True, block=32):
    """查询 × 候选的对数似然，与先验拟合、锚点更新同一观测模型。
    未检出 AP：log(1 − h·Φ((μ−θ)/√(s²+v)))；检出 AP：数值项 N(x; μ+c, σ²+v)（扫描偏移 c 解析积分）乘检出项。
    exact=True：检出项取给定 x 后 ℓ 的后验，h·Φ((ℓ*−θ)/√(s²+v*))，ℓ* = μ + v/(v+σ²)·(x−μ)，v* = vσ²/(v+σ²)
    （这一项里 c 取 0）。exact=False：把检出与数值当作独立的乘积近似 h·Φ((μ−θ)/√(s²+v))，仅作消融。"""
    sig2, tau2 = obs["sigma"] ** 2, obs["tau"] ** 2
    h, theta, s2 = tf["h"], tf["theta"], tf["s"] ** 2
    aps = np.flatnonzero(present)
    mu, vlat, q = mu[:, aps], vlat[:, aps], q_rssi[:, aps]
    d = (q != 100).astype(np.float64)
    x = np.where(q != 100, q, 0).astype(np.float64)
    v = sig2 + vlat
    iv = 1 / v
    s0 = d @ iv.T
    s1 = x @ iv.T - d @ (mu * iv).T
    sq = (x * x) @ iv.T - 2 * x @ (mu * iv).T + d @ (mu * mu * iv).T
    z = (mu - theta) / np.sqrt(s2 + vlat)
    ll = (1 - d) @ np.log1p(-np.minimum(h * ndtr(z), 1 - 1e-12)).T \
        - 0.5 * (sq - tau2 * s1 ** 2 / (1 + tau2 * s0)) - 0.5 * (d @ np.log(v).T + np.log1p(tau2 * s0))
    if not exact:
        return ll + d @ (np.log(h) + log_ndtr(z)).T
    w, sd = vlat / v, np.sqrt(s2 + vlat * sig2 / v)
    for i in range(0, len(q), block):
        xb, db = x[i:i + block, None], d[i:i + block, None]
        lstar = mu[None] + w[None] * (xb - mu[None])
        ll[i:i + block] += (db * log_ndtr((lstar - theta) / sd[None])).sum(-1) + db.sum(-1) * np.log(h)
    return ll
