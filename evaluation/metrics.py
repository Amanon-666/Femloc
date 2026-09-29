"""按论文Eq.(20)报告逐扫描MDE及固定适应步数曲线。"""
import numpy as np


def evaluate(predictions, coordinates):
    distances = np.linalg.norm(predictions - coordinates[None, :, :], axis=-1)
    return [{"step": step, "mde": float(errors.mean()), "median": float(np.median(errors))}
            for step, errors in enumerate(distances)]


def first_hit(curve, threshold):
    return next((row["step"] for row in curve if row["mde"] <= threshold), None)
