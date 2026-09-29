"""在真实 UJI 源楼层检查跨设备 episode 的信息权限。"""

import argparse
import json
from pathlib import Path

import numpy as np

from data.episodes import (prepare_floor, sample_cross_device_episode,
                           sample_paired_device_episode)
from data.uji import load_floors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cross_device.json"))
    config = json.loads(parser.parse_args().config.read_text())
    raw = load_floors(config["data_path"])
    excluded = set(config["development_targets"] + config["confirmation_targets"])
    rng = np.random.default_rng(0)
    for name in sorted(set(raw) - excluded):
        floor = prepare_floor(raw[name], "cpu")
        sampler = sample_paired_device_episode if config.get("paired_device_source") else sample_cross_device_episode
        support, query = sampler(floor, rng, 10, 3, 5)
        support_phones = set(raw[name]["phones"][support])
        query_phones = set(raw[name]["phones"][query])
        assert len(support) == 30 and len(query) == 50
        assert len(support_phones) == len(query_phones) == 1
        assert support_phones.isdisjoint(query_phones)
        assert set(floor["position_keys"][i] for i in support).isdisjoint(
            floor["position_keys"][i] for i in query)
        assert set(floor["group_ids"][support]).isdisjoint(floor["group_ids"][query])
        if config.get("paired_device_source"):
            both = floor["phone_positions"]
            assert all(floor["position_keys"][i] in both[next(iter(query_phones))] for i in support)
            assert all(floor["position_keys"][i] in both[next(iter(support_phones))] for i in query)
        print(name, "passed")


if __name__ == "__main__":
    main()
