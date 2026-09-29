# DKT 首轮状态

官方类源码与GP后验公式核验通过，见DKT_CHECK.json。
配置configs/dkt_uji.json：三个seed各5000次源任务更新，每目标20个30扫描episode；目标参数更新为0。
运行目录outputs/dkt_v1，日志outputs/dkt_v1.log；最终以结果完整性和completed.json核验为准。
本轮启动后自动值守；不按目标结果调参，不自动重试失败运行。
