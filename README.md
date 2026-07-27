# 福彩推荐系统

> 面向彩票店老板的号码推荐工具 — 用数据说话，给顾客"有理论依据"的推荐。

**不是预测中大奖的机器。** 彩票本质随机，任何算法都无法保证中一等奖。

**是**帮你：
- 给顾客提供四种不同风格的号码推荐
- 用历史数据建立专业形象
- 避免"今天买啥"的尴尬

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证安装
python -m pytest tests/test_core.py -v

# 3. 生成今日推荐
python scripts/generate_recommendation.py
```

---

## Docker 部署（Windows）

```powershell
# 构建并启动
docker-compose up -d

# 浏览器访问
http://localhost:8000
# 默认账号: admin / caipiao2026
```

---

## 项目结构

```
lottery-predictor/
├── src/
│   ├── api.py              # FastAPI Web 服务
│   ├── recommendation.py   # 多策略推荐引擎
│   ├── unified_pipeline.py # ML 训练/预测管线
│   ├── feature_engineering.py  # 特征工程
│   ├── modeling.py         # XGBoost 基学习器
│   ├── model_lstm.py       # LSTM 基学习器
│   ├── model_poisson.py    # 泊松先验基学习器
│   ├── model_stacking.py   # Stacking 元学习器
│   ├── backtest.py         # 回测引擎
│   ├── strategy_backtest.py    # 策略回测
│   ├── analysis.py         # 分析报告生成
│   ├── user_history.py     # 用户历史记录
│   ├── data_fetcher.py     # 数据加载
│   └── config.py           # 全局配置
├── static/
│   ├── index.html          # Web 前端（单页应用）
│   └── login.html          # 登录页
├── scripts/
│   ├── generate_recommendation.py  # 一键生成推荐
│   ├── strategy_ranking.py         # 策略排行榜
│   ├── train.py                    # 模型训练
│   └── predict.py                  # 号码预测
├── data/                   # 历史数据
├── tests/
│   └── test_core.py        # 单元测试（20 个）
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── docs/
    ├── user_manual.md      # 用户手册
    ├── design.md           # 设计文档
    └── architecture.md     # 架构文档
```

---

## 四种推荐策略

| 策略 | 思路 | 适合顾客 |
|------|------|----------|
| **保守型** | 选近期出现次数最多的热号 | 相信"热者恒大"的老顾客 |
| **激进型** | 选遗漏最大、最久没出的冷号 | 认为"该出了"的顾客 |
| **平衡型** | 兼顾和值、奇偶、区间均衡 | 追求"看起来合理"的顾客 |
| **玄学型** | 用幸运数字组合 | 有自己幸运数字的顾客 |

---

## 支持的彩种

| 代码 | 彩种 | 数据量 | 状态 |
|------|------|--------|------|
| `ssq` | 双色球 | 3277 期 (2003-2025) | ✅ |
| `sd` | 福彩3D | 7703 期 (2004-2026) | ✅ |
| `qlc` | 七乐彩 | 2964 期 (2007-2026) | ✅ |

---

## API 接口

```
POST   /api/login             登录
POST   /api/logout            退出
GET    /api/me                当前状态
GET    /api/recommend/{code}  四种策略推荐
GET    /api/predict/{code}    ML 预测（支持 ?method=xgb/lstm/poisson）
GET    /api/history/{code}    历史数据（支持 ?limit=30）
GET    /api/stats/{code}      统计图表数据
GET    /api/ranking/{code}    策略排行榜
GET    /api/report/{code}     综合报告
```

---

## 命令行工具

```bash
# 生成推荐
python scripts/generate_recommendation.py

# 策略排行榜
python scripts/strategy_ranking.py --window 200 --backtest 50

# 训练模型
python scripts/train.py --name ssq --method xgb

# 预测下一期
python scripts/predict.py --name ssq --method xgb
```

---

## 技术栈

- **后端**: FastAPI + Uvicorn
- **前端**: 原生 HTML + ECharts（零构建工具）
- **ML**: XGBoost + LSTM + Poisson + Stacking
- **部署**: Docker + Docker Compose

---

## 免责声明

- 本系统**不保证中奖**
- 所有推荐基于历史数据统计，仅供娱乐参考
- 请理性购彩，量力而行
- 未成年人不得购买彩票

---

*最后更新：2026-07-27*
