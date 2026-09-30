"""TPM 跨楼层先验作为表格上下文学习（TabICLv2）的合成上下文；推导见 docs/research/ICL_CONTEXT_DERIVATION.md。

合成扫描按 TPM 先验预测分布采样：每条扫描、每个 AP 独立抽潜在水平 ℓ ~ N(μ, v)，
检出 ~ Bernoulli(h·Φ((ℓ−θ)/s))，检出值 = round(ℓ + σ_t·ε)。上下文模型只做前向推断，不训练。
"""
import sys

import numpy as np
from scipy.special import ndtr

from models.cross_floor import CEN, ap_bin

TABICL_DEPS = "/home/panyushuo/projects/panyushuo/RNP-UJI/.deps_tabicl"
CHECKPOINT = "/home/panyushuo/projects/panyushuo/RNP-UJI/tabicl_weights/tabicl-regressor-v2-20260212.ckpt"
NO_DETECT = 100
SCANS_PER_POSITION = 3


def latent_prior(lc, tf):
    """与 prior_map 相同的先验：返回旧层每个坐标、每个 AP 的潜在水平均值与方差（旧层从未检出的 AP 记为不存在）。"""
    st = tf["sigma_total"] ** 2
    kL = lc["k"]
    det = kL >= 1
    zb = np.broadcast_to(ap_bin(lc), kL.shape)
    rho = tf["rho"][zb]
    mu = np.where(det, CEN + rho * (lc["s1"] / np.maximum(kL, 1) - CEN) + tf["bbar"][zb], tf["mu0"][zb])
    v = tf["sb2"][zb] + tf["s_w2"] + np.where(det, rho ** 2 * st / np.maximum(kL, 1), tf["vL0"][zb])
    return mu, v, lc["k"].sum(0) > 0


def sample_prior_scans(lc, tf, rng, scans=SCANS_PER_POSITION):
    """每个旧层坐标 scans 条合成扫描（UJI 编码：检出为整数 dBm，未检出为 100）。"""
    mu, v, present = latent_prior(lc, tf)
    mu, v = np.repeat(mu, scans, 0), np.repeat(v, scans, 0)
    level = mu + np.sqrt(v) * rng.standard_normal(mu.shape)
    detected = rng.random(mu.shape) < tf["h"] * ndtr((level - tf["theta"]) / tf["s"])
    value = np.clip(np.round(level + tf["sigma_total"] * rng.standard_normal(mu.shape)), -104, 0)
    rssi = np.where(detected & present[None], value, NO_DETECT).astype(np.int16)
    return rssi, np.repeat(lc["xy"], scans, 0)


def encode(rssi):
    """与 RNP-UJI 的 TabICLv2 基线相同：检出 (x+105)/105，未报告 0。"""
    return np.where(rssi == NO_DETECT, 0.0, (rssi.astype(np.float64) + 105.0) / 105.0)


class ContextRegressor:
    """两轴分别用 TabICLv2 回归；坐标相对 origin。"""

    def __init__(self, device="cuda:0", n_jobs=4):
        if TABICL_DEPS not in sys.path:
            sys.path.insert(0, TABICL_DEPS)
        from tabicl import TabICLRegressor
        self._make = lambda axis, seed: TabICLRegressor(model_path=CHECKPOINT, allow_auto_download=False,
                                                        device=device, n_jobs=n_jobs, random_state=seed * 2 + axis)

    def predict(self, ctx_rssi, ctx_xy, queries, origin, ctx_flag=None, seed=0):
        """queries: {名称: RSSI}。ctx_flag 为每行来源标记（合成 0、锚点 1），查询统一标记 1；None 表示不加标记列。"""
        X = encode(ctx_rssi)
        keep = X.std(0) > 0
        X = X[:, keep]
        qs = {k: encode(q)[:, keep] for k, q in queries.items()}
        if ctx_flag is not None:
            X = np.column_stack([X, ctx_flag])
            qs = {k: np.column_stack([q, np.ones(len(q))]) for k, q in qs.items()}
        names = list(qs)
        stacked = np.vstack([qs[k] for k in names])
        y = ctx_xy - origin
        pred = np.column_stack([self._make(a, seed).fit(X, y[:, a]).predict(stacked) for a in range(2)]) + origin
        out, i = {}, 0
        for k in names:
            out[k] = pred[i:i + len(qs[k])]
            i += len(qs[k])
        return out, int(keep.sum())
