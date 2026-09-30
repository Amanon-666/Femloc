"""直接复用 GUFU 的 RSSI 边权转换。

Source: khchiuac/GUFU, commit e1f14c73165daceead6992b5047507704fb0d7c5,
util.py::rssi2weight. 参数 offset 保持原作者接口。
"""


def rssi2weight(offset, rssi):
    return offset + rssi
