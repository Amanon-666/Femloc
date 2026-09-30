"""Fixed-association fingerprint reconstruction; derivation in docs/research/.

The public update consumes RSSI only for unlabeled samples. It never estimates
hyperparameters or sees query samples. This is GUFU-inspired, not GUFU itself.
"""
import numpy as np
from scipy.linalg import eigh, solve

from scripts.evaluate_scm_tobit import tobit_weights
from scripts.evaluate_signal_calibrated_map import dbm
from scripts.evaluate_transfer_map import calibrate


def conditional_factor(positions, anchors, length=40.0, noise=0.3):
    def kernel(a, b):
        return np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / (2 * length**2))
    kps = kernel(positions, anchors)
    c = kernel(positions, positions) - kps @ solve(
        kernel(anchors, anchors) + noise * np.eye(len(anchors)), kps.T,
        assume_a="pos")
    vals, vecs = eigh((c + c.T) / 2)
    keep = vals > max(float(vals[-1]), 1e-15) * 1e-10
    return vecs[:, keep] * np.sqrt(vals[keep])


def reconstruct(f0, x, h, factor, strength, anchor_count, noise=0.3):
    """Quadratic update for a fixed association; also used by oracle diagnostics.

    h is a row-stochastic association, not a table of true location labels.
    The returned map is clipped, but the stationarity check is pre-clipping.
    """
    if strength < 0:
        raise ValueError("strength must be nonnegative")
    if strength == 0 or len(x) == 0:
        return f0.copy(), {"update_rms": 0.0, "rank": int(factor.shape[1])}
    if h.shape != (len(x), len(f0)) or np.any(h < 0) or not np.allclose(h.sum(1), 1):
        raise ValueError("invalid association matrix")
    tau = strength * anchor_count / (noise * len(x))
    a = h @ factor
    residual = x - h @ f0
    lhs = np.eye(a.shape[1]) + tau * a.T @ a
    rhs = tau * a.T @ residual
    b = solve(lhs, rhs, assume_a="pos")
    delta = factor @ b
    virtual = np.clip(f0 + delta, -110.0, 0.0)
    before = float(np.mean(residual**2))
    after = float(np.mean((x - h @ virtual)**2))
    normal_error = float(np.linalg.norm(lhs @ b - rhs) / max(np.linalg.norm(rhs), 1e-12))
    return virtual, {
        "rank": int(factor.shape[1]), "update_rms": float(np.sqrt(np.mean((virtual - f0)**2))),
        "reconstruction_mse_before": before, "reconstruction_mse_after_clip": after,
        "objective_before": float(tau / 2 * np.sum(residual**2)),
        "objective_after_unclipped": float((np.sum(b**2) + tau * np.sum((residual - a @ b)**2)) / 2),
        "clipped_fraction": float(np.mean(virtual != f0 + delta)),
        "normal_equation_relative_error": normal_error,
    }


def prepare_update(base, support_rssi, support_xy, unlabeled_rssi,
                   field=(40.0, 0.3), match=(12.0, -85.0)):
    calibrated, positions = calibrate(base, support_rssi, support_xy, field)
    m = len(base[0])
    # Only original map nodes are updated; support prototypes are kept intact.
    f0 = calibrated[:m]
    anchors = np.unique(support_xy, axis=0)
    h = tobit_weights(unlabeled_rssi, f0, *match)
    return {"f0": f0, "positions": positions, "support_prototypes": calibrated[m:],
            "x": dbm(unlabeled_rssi), "h": h,
            "factor": conditional_factor(positions[:m], anchors, *field),
            "anchor_count": len(anchors), "noise": field[1]}


def apply_update(prepared, strength):
    virtual, diagnostics = reconstruct(
        prepared["f0"], prepared["x"], prepared["h"], prepared["factor"],
        strength, prepared["anchor_count"], prepared["noise"])
    return (np.vstack([virtual, prepared["support_prototypes"]]), prepared["positions"]), diagnostics
