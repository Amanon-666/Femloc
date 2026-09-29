"""在真实 UJI 源楼层检查跨设备 episode 的信息权限。"""

import json
from pathlib import Path

import numpy as np

from data.episodes import prepare_floor, sample_cross_device_episode
from data.uji import load_floors


def main():
    config = json.loads(Path("configs/cross_device.json").read_text())
    raw = load_floors(config["data_path"])
    excluded = set(config["development_targets"] + config["confirmation_targets"])
    rng = np.random.default_rng(0)
    for name in sorted(set(raw) - excluded):
        floor = prepare_floor(raw[name], "cpu")
        support, query = sample_cross_device_episode(floor, rng, 10, 3, 5)
        support_phones = set(raw[name]["phones"][support])
        query_phones = set(raw[name]["phones"][query])
        assert len(support) == 30 and len(query) == 50
        assert len(support_phones) == len(query_phones) == 1
        assert support_phones.isdisjoint(query_phones)
        assert set(floor["position_keys"][i] for i in support).isdisjoint(
            floor["position_keys"][i] for i in query)
        assert set(floor["group_ids"][support]).isdisjoint(floor["group_ids"][query])
        print(name, "passed")


if __name__ == "__main__":
    main()
