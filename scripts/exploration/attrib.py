"""误差归因（仅诊断，不用于选参）：把 SCM 虚拟地图中某一类 (位置, AP) 单元换成上层真实值，看定位误差降多少。
类 1：旧层同位置可靠检出（q≥0.5）；类 2a：旧层同位置未可靠检出，但该 AP 在旧层别处可靠检出；类 2b：该 AP 在旧层任何位置都未可靠检出。"""
import numpy as np
from data.uji import load_floors
from models.radio_map import cells
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import build_map, episodes, lower_map, mde
R = "/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/"
tr, va = load_floors(R + "trainingData.csv"), load_floors(R + "validationData.csv")
CFG = {"anchor_lookup_neighbors": 3}
rows = []
for lo, up in PAIRS:
    Lp, Lxy = lower_map(tr[lo]); Up, Uxy = lower_map(tr[up])
    lc = cells(tr[lo]); assert np.allclose(lc["xy"], Lxy)
    qL = lc["k"] / lc["n"][:, None]
    d = np.sqrt(((Uxy[:, None] - Lxy[None]) ** 2).sum(-1)); keep = d.min(1) <= 1.0
    U, J = np.flatnonzero(keep), d.argmin(1)[keep]
    r1 = qL[J] >= 0.5
    apL = (qL >= 0.5).any(0)
    r2a, r2b = ~r1 & apL[None], ~r1 & ~apL[None]
    T, O, pos = Up[U], Lp[J], Uxy[U]
    sig = T > -109.9
    share = [float((m & sig).sum() / sig.sum()) for m in (r1, r2a, r2b)]
    vq, vy = va[up]["rssi"], va[up]["xy"]
    f = lambda M: mde(tobit_match(vq, M, pos, 16.0, -80.0), vy)
    res = {"TRUE": f(T), "OLD": f(O)}
    rng, acc = np.random.default_rng(1001), {}
    for s, _ in episodes(tr[up]["xy"], 10, 5, 3, rng):
        V = build_map("scm", (Lp, Lxy), tr[up]["rssi"][s], tr[up]["xy"][s], (60.0, 0.3), CFG)[0][:len(Lxy)][J]
        for name, M in (("SCM", V), ("true_on_1", np.where(r1, T, V)), ("true_on_2a", np.where(r2a, T, V)),
                        ("true_on_2b", np.where(r2b, T, V)), ("true_on_2", np.where(r1, V, T))):
            acc.setdefault(name, []).append(f(M))
    res.update({k: float(np.mean(v)) for k, v in acc.items()})
    rows.append(res)
    print(f"{up} pos={len(U):3d} signal-share r1/r2a/r2b={np.round(share, 2)} " + " ".join(f"{k}={v:.2f}" for k, v in res.items()), flush=True)
print("EQUAL " + " ".join(f"{k}={np.mean([r[k] for r in rows]):.2f}" for k in rows[0]))
