---
name: multi-lottery-architecture
description: 多玩法彩票系统架构状态和关键设计决策
type: project
---

## 多玩法彩票系统架构

**Why:** 用户要求从单一双色球扩展到4种彩票（双色球、福彩3D、七乐彩、快乐8）
**How to apply:** 所有新功能开发需遵循多玩法架构，使用LotteryModelConfig配置驱动

### 已完成的关键修复

1. **前端加载卡死** (P0): `loadRecommend/loadHistory/loadStats` 添加 try-catch + 错误提示+重试链接
2. **QLC解析修复** (P0): `_parse_issue_list` qlc分支从错误的多td访问改为解析`td[1]`中的`<span class="cBlue">`获取蓝球+红球文本
3. **QLC数据格式** (P0): 重新下载后自动使用正确列名（期数/红球_1-7/蓝球_1/开奖日期）
4. **增量数据更新** (P1): `download_history(merge=True)` 支持增量合并，无新数据时保留原有数据不覆盖

### 500.com QLC页面HTML结构

- `table#tablelist`，6列：期号/开奖号码/和值/销量/奖池/开奖日期
- 开奖号码在`td[1]`：7个红球为纯文本 + 1个蓝球在`<span class="cBlue">`
- 不同于SSQ每个球单独td的结构

### 前端球类型 (ballType)

- `red-blue`: 双色球（6红+1蓝）
- `three`: 福彩3D（百位/十位/个位）
- `red-special`: 七乐彩（7红+1特别号）- 待实现
- `pick-n`: 快乐8（选号模式）- 待实现

### 待办清单状态

- P0/P1关键修复: 全部完成
- P1-P5架构扩展: 30项待办（见todo list）