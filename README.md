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

# 2. 配置环境变量（必须！）
cp .env.example .env
# 编辑 .env 文件，修改 LOTTERY_PASS 和 LOTTERY_SECRET

# 3. 运行测试
python -m pytest tests/ -v

# 4. 生成今日推荐
python scripts/generate_recommendation.py
```

---

## Docker 部署（Windows）

```powershell
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填写实际密码

# 2. 构建并启动
docker-compose up -d --build

# 3. 浏览器访问
http://localhost:8000
# 默认账号: admin / （在 .env 中设置的密码）
```

---

## 项目结构

```
welfare_predict/
├── src/
│   ├── api.py              # FastAPI Web 服务（v2.1: SQLite 会话/CORS/健康检查）
│   ├── session.py          # SQLite 持久化会话管理（P0-02 新增）
│   ├── bootstrap.py        # 应用启动引导 / 日志配置（P3-02 新增）
│   ├── model_io.py         # 统一模型序列化接口（P1-04 新增）
│   ├── recommendation.py   # 多策略推荐引擎（四种策略）
│   ├── unified_pipeline.py # ML 训练/预测管线（含完整 Stacking 实现）
│   ├── feature_engineering.py  # 特征工程（已优化 O(n²)→O(n)）
│   ├── modeling.py         # XGBoost 基学习器
│   ├── model_lstm.py       # MLP/DNN 基学习器（P1-02 重构：原 LSTM 已改为 MLP）
│   ├── model_poisson.py    # 泊松先验基学习器
│   ├── model_stacking.py   # Stacking 元学习器
│   ├── backtest.py         # 回测引擎（P4-03: 魔法数字已提取为常量）
│   ├── strategy_backtest.py    # 策略回测对比
│   ├── analysis.py         # 综合分析报告生成
│   ├── user_history.py     # 用户历史记录
│   ├── data_fetcher.py     # 数据抓取（P2-01: KL8 已清理）
│   ├── config.py           # 全局配置（P2-01: KL8 已移除）
│   └── common.py           # 公共工具函数
├── static/
│   ├── index.html          # Web 前端（单页应用）
│   └── login.html          # 登录页
├── scripts/
│   ├── generate_recommendation.py  # 一键生成推荐
│   ├── strategy_ranking.py         # 策略排行榜
│   ├── train.py                    # 模型训练
│   ├── predict.py                  # 号码预测
│   ├── get_data.py                 # 数据下载
│   ├── parse_lotterydata.py        # 数据解析
│   └── import_csv.py               # CSV 导入
├── tests/
│   ├── conftest.py          # 测试配置（P0-04: 已修复崩溃问题）
│   ├── test_core.py         # 核心算法测试
│   ├── test_preprocessing.py    # 预处理测试
│   ├── test_modeling.py         # 模型测试
│   ├── test_pipeline.py         # 管线测试
│   ├── test_config.py           # 配置测试
│   ├── test_api.py              # API 端到端测试（P2-04 新增）
│   └── test_backtest.py         # 回测引擎测试（P2-05 新增）
├── config/
│   └── config.yaml          # 运行时配置
├── data/                   # 历史数据（git ignored）
├── model/                  # 训练模型（git ignored）
├── users/                  # 用户数据（git ignored）
├── Dockerfile              # 镜像构建（P0-01/P3-04 已优化）
├── docker-compose.yml      # 编排文件（P0-01 已移除硬编码密码）
├── .dockerignore           # 构建排除（P3-04 新增）
├── .env.example            # 环境变量模板（P4-05 新增）
├── .gitignore              # Git 忽略规则（已更新）
├── requirements.txt        # Python 依赖（P0-03: 补齐 xgboost/joblib）
├── Makefile                # 快捷命令
└── docs/
    ├── user_manual.md      # 用户手册
    ├── design.md           # 设计文档
    └── architecture.md     # 架构文档
```

---

## 四种推荐策略

| 策略 | 思路 | 适合顾客 |
|------|------|----------|
| **保守型** | 选近期出现次数最多的热号 | 相信"热者恒热"的老顾客 |
| **激进型** | 选遗漏最大、最久没出的冷号 | 认为"该出了"的顾客 |
| **平衡型** | 兼顾和值、奇偶、区间均衡 | 追求"看起来合理"的顾客 |
| **玄学型** | 用幸运数字组合 | 有自己幸运数字的顾客 |

---

## 支持的彩种

| 代码 | 彩种 | 数据量 | 状态 |
|------|------|--------|------|
| `ssq` | 双色球 | 3277+ 期 (2003-2025) | ✅ |
| `dlt` | 大乐透 | — | ✅ |
| `sd` | 福彩3D | 7703+ 期 (2004-2026) | ✅ |
| `qlc` | 七乐彩 | 2964+ 期 (2007-2026) | ✅ |
| `pls` | 排列三 | — | ✅ |
| `qxc` | 七星彩 | — | ✅ |
| ~~`kl8`~~ | ~~快乐8~~ | — | ❌ 已迁移至独立项目 (2025-10) |

---

## API 接口

```text
POST   /api/login             登录（SQLite 持久化会话）
POST   /api/logout            退出
GET    /api/me                当前登录状态
GET    /health                健康检查端点（P3-03 新增）
GET    /api/recommend/{code}  四种策略推荐
GET    /api/predict/{code}    ML 预测（支持 ?method=xgb/lstm/poisson/stacking）
GET    /api/history/{code}    历史数据（支持 ?limit=30，范围 [1,200]）
GET    /api/stats/{code}      统计图表数据（热冷号/遗漏）
GET    /api/ranking/{code}    策略排行榜
GET    /api/report/{code}     综合报告
```

### 安全改进（v2.1）

- ✅ CORS 跨域限制
- ✅ 全局异常处理器（隐藏内部路径）
- ✅ 参数边界校验（limit, window, backtest）
- ✅ SQLite 持久化会话（支持多用户）
- ✅ 硬编码密码已移除（必须通过 .env 注入）

---

## 命令行工具

```bash
# 生成推荐
python scripts/generate_recommendation.py

# 策略排行榜
python scripts/strategy_ranking.py --window 200 --backtest 50

# 训练模型（支持 xgb/lstm/poisson/stacking）
python scripts/train.py --name ssq --method xgb
python scripts/train.py --name ssq --method stacking  # 完整 Stacking 流程

# 预测下一期
python scripts/predict.py --name ssq --method xgb

# 下载数据
python scripts/get_data.py --code ssq
```

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI + Uvicorn | 异步 API 服务 |
| **前端** | 原生 HTML + ECharts | 零构建工具 |
| **ML 核心** | XGBoost + MLP(DNN) + Poisson + Stacking | 四种基学习器 + 元学习器 |
| **特征工程** | NumPy/Pandas 向量化 | 已优化性能热点 |
| **会话存储** | SQLite（session.py） | 替代原内存单用户模式 |
| **序列化** | ModelIO 统一接口 | joblib + TF native 自动分发 |
| **部署** | Docker + Docker Compose | 含 healthcheck |
| **日志** | loguru + config.yaml | 自动读取日志配置 |

---

## v2.1 更新记录（2026-07-27）

### 🔴 P0 — 安全/稳定性修复
- 移除 Dockerfile/docker-compose.yml 中的硬编码密码
- 重构会话管理：内存单用户 → SQLite 持久化多用户
- 补齐 requirements.txt 缺失的 xgboost/joblib 依赖
- 修复 conftest.py 引用不存在属性导致的测试崩溃

### 🟠 P1 — 架构/正确性修复
- 实现完整的 Stacking ensemble 流程（不再静默降级）
- LSTM 重构为 MLP/DNN（修正 reshape 导致的时序信息丢失）
- 优化 O(n²) 性能热点（滑动窗口 Counter + 向量化蓝球选择）
- 统一模型序列化方式（ModelIO 接口）

### 🟡 P2 — 健壮性改进
- 清理 KL8 已迁移残留代码（config/data_fetcher）
- 添加 API 安全中间件（CORS/参数校验/全局异常处理）
- 修复类型注解不规范（any → Any）
- 补充 API 层和回测引擎单元测试

### 🟢 P3 — 性能/体验优化
- 质合特征计算向量化（NumPy 替代 Python 循环）
- loguru 日志配置生效（bootstrap.py）
- Docker 镜像瘦身（.dockerignore + COPY 优化）
- 添加 /health 健康检查端点

### 🔵 P4 — 规范/文档
- 提取魔法数字为命名常量（PRIZE_MONEY 等）
- 添加 .env.example 环境变量模板
- 更新项目结构和文档

---

## 免责声明

- 本系统**不保证中奖**
- 所有推荐基于历史数据统计，仅供娱乐参考
- 请理性购彩，量力而行
- 未成年人不得购买彩票

---

*最后更新：2026-07-27 | 版本：v2.1*
