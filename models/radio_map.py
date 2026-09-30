"""潜在无线电地图的观测模型：检出与数值分开建模（hurdle）。

UJI 的 100 表示本次扫描没有报告该 AP，不是 −110 dBm 的测量值。弱 AP 在同一位置也只被部分扫描报告
（检出率随信号平滑下降，强信号仍有约 6% 随机漏检）。每个 (位置, AP) 单元用检出概率 p 和
检出时的数值分布 N(m, v) 描述；设备/用户偏移 c ~ N(0, τ²) 作用于整条扫描，在似然中解析积分掉。
所有参数只由源楼层 trainingData 按生成式估计，不用定位误差调参。
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

NO_DETECT = 100


def cells(floor):
    """去重后按坐标聚合：n 每位置扫描数，k/s1/s2 为每 AP 检出次数与检出值的一、二阶和。"""
    _, first = np.unique(floor["groups"], return_index=True)
    rssi, xy = floor["rssi"][first], floor["xy"][first]
    keys, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = inv.ravel()
    det = rssi != NO_DETECT
    x = np.where(det, rssi, 0).astype(np.float64)
    k, s1, s2 = (np.zeros((len(keys), rssi.shape[1])) for _ in range(3))
    np.add.at(k, inv, det)
    np.add.at(s1, inv, x)
    np.add.at(s2, inv, x * x)
    return {"xy": keys, "n": np.bincount(inv).astype(np.float64), "k": k, "s1": s1, "s2": s2,
            "rssi": rssi, "inv": inv}


def fit_observation(floor_cells):
    """σ_t：单元内检出值的合并标准差，含扫描偏移；τ：整条扫描的共同偏移（设备/用户）；
    σ = sqrt(σ_t² − τ²)：逐 AP 独立噪声，避免 τ 在似然里被重复计入；
    (α,β)：检出次数的 beta-binomial 先验；low：几乎检不到的单元（1–2 次检出）的数值分布。"""
    num = sum(float((c["s2"] - c["s1"] ** 2 / np.maximum(c["k"], 1))[c["k"] >= 2].sum()) for c in floor_cells)
    den = sum(float((c["k"] - 1)[c["k"] >= 2].sum()) for c in floor_cells)
    sigma_t = np.sqrt(num / den)
    rbar, noise, kn, low = [], [], [], []
    for c in floor_cells:
        K, S1, X = c["k"][c["inv"]], c["s1"][c["inv"]], c["rssi"].astype(np.float64)
        D = (c["rssi"] != NO_DETECT) & (K >= 5)
        r = np.where(D, X - (S1 - X) / np.maximum(K - 1, 1), 0.0)
        cnt = D.sum(1)
        ok = cnt >= 5
        rbar.append(r[ok].sum(1) / cnt[ok])
        noise.append(np.where(D, 1 + 1 / np.maximum(K - 1, 1), 0).sum(1)[ok] / cnt[ok] ** 2)
        present = c["k"].sum(0) > 0
        kn.append(np.stack([c["k"][:, present].ravel(), np.repeat(c["n"], present.sum())], 1))
        weak = (c["k"] >= 1) & (c["k"] <= 2) & (c["n"][:, None] >= 10)
        low.append((c["s1"] / np.maximum(c["k"], 1))[weak])
    rbar, noise, low = np.concatenate(rbar), np.concatenate(noise), np.concatenate(low)
    # var(r̄) = τ² + σ²·A，σ² = σ_t² − τ²，A 为逐扫描独立噪声的平均系数 → 解出 τ²
    A = noise.mean()
    tau2 = max((rbar.var() - sigma_t ** 2 * A) / (1 - A), 0.0)
    tau, sigma = np.sqrt(tau2), np.sqrt(sigma_t ** 2 - tau2)
    u, cnt = np.unique(np.concatenate(kn), axis=0, return_counts=True)
    nll = lambda t: -(cnt * (betaln(u[:, 0] + np.exp(t[0]), u[:, 1] - u[:, 0] + np.exp(t[1])) - betaln(*np.exp(t)))).sum()
    a, b = np.exp(minimize(nll, [np.log(0.3), np.log(3.0)], method="Nelder-Mead").x)
    # h：强信号单元（检出均值 > −60 dBm）的合并检出率 = 与楼层无关的随机漏检之外的上限
    strong = [(c["k"][(c["s1"] / np.maximum(c["k"], 1) > -60) & (c["k"] >= 1) & (c["n"][:, None] >= 10)],
               np.broadcast_to(c["n"][:, None], c["k"].shape)[(c["s1"] / np.maximum(c["k"], 1) > -60) & (c["k"] >= 1)
                                                               & (c["n"][:, None] >= 10)]) for c in floor_cells]
    h_det = sum(k.sum() for k, _ in strong) / sum(n.sum() for _, n in strong)
    return {"sigma": float(sigma), "sigma_total": float(sigma_t), "tau": float(tau), "h_det": float(h_det),
            "alpha": float(a), "beta": float(b),
            "low_mean": float(low.mean()), "low_sd": float(low.std()), "n_scans_tau": int(len(rbar))}


def cell_predictive(c, obs):
    """实测地图（该层 trainingData）每单元的检出概率与检出值分布。"""
    n, k = c["n"][:, None], c["k"]
    p = (k + obs["alpha"]) / (n + obs["alpha"] + obs["beta"])
    seen = k > 0
    m = np.where(seen, c["s1"] / np.maximum(k, 1), obs["low_mean"])
    v = np.where(seen, obs["sigma"] ** 2 + obs["sigma_total"] ** 2 / np.maximum(k, 1), obs["sigma"] ** 2 + obs["low_sd"] ** 2)
    return p, m, v, k.sum(0) > 0


def loglik(q_rssi, p, m, v, present, tau):
    """查询 × 候选的对数似然。检出项 log p + 高斯数值项（共同偏移 c 用 Sherman–Morrison 积分），
    未检出项 log(1−p)。只用该地图上出现过的 AP；对所有候选相同的常数项省略。"""
    hit = q_rssi != NO_DETECT
    d = (hit & present).astype(np.float64)
    u = (~hit & present).astype(np.float64)
    x = np.where(hit & present, q_rssi, 0).astype(np.float64)
    iv = 1.0 / v
    ll = d @ np.log(p).T + u @ np.log1p(-p).T
    s0 = d @ iv.T
    s1 = x @ iv.T - d @ (m * iv).T
    s2 = (x * x) @ iv.T - 2 * x @ (m * iv).T + d @ (m * m * iv).T
    t2 = tau ** 2
    return ll - 0.5 * (s2 - t2 * s1 ** 2 / (1 + t2 * s0)) - 0.5 * (d @ np.log(v).T + np.log1p(t2 * s0))


def posterior_mean(ll, xy, temperature=1.0):
    s = ll / temperature
    w = np.exp(s - s.max(1, keepdims=True))
    return (w / w.sum(1, keepdims=True)) @ xy
