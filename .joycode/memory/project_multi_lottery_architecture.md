---
name: multi-lottery-architecture
description: 多玩法彩票数据源架构和KL8趋势图解析要点
type: project
---

KL8数据源: datachart.500.com/kl8/ 趋势图(非标准history.shtml)。解析关键: td含chartBall01类=当期开奖号(20个)，不可用text=="1"(遗漏值1=上期出现)。无日期列，用"最新期≈当前日期"参考点推算。SD修复: _repair_sd_data()从3d/data.csv重建含试机号数据，期数必须为str。域名白名单: datachart.500.com和data.917500.cn(kaijiang.500.com不可访问)。