# 福彩推荐系统 — 技术文档

> 本文档为 [README.md](README.md) 的技术附录，包含架构、API、部署等技术细节。

---

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│                    用户界面层                         │
│  index.html (ECharts)  │  login.html  │  REST API   │
└────────────┬───────────┴──────┬───────┴──────┬───────┘
             │                  │              │
┌────────────┼──────────────────┼──────────────┼───────┐
│            ▼                  ▼              ▼       │
│  ┌──────────────────────────────────────────────┐   │
│  │         API 服务层 (src/api.py)              │   │
│  │  CORS 中间件 │ 参数校验 │ 全局异常处理器     │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                           │
│  ┌──────────────────────▼───────────────────────┐   │
│  │       会话管理层 (src/session.py)            │   │
│  │  SQLite 持久化 / 多用户 / 过期清理           │   │
│  └──────────────────────┬───────────────────────┘   │
└─────────────────────────┼───────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────┐
│                         ▼                           │
│  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │ Recommendation   │  │ UnifiedPipeline          │  │
│  │ Engine (四策略)  │  │ (训练/预测管线)          │  │
│  └────────┬────────┘  └──────────┬───────────────┘  │
│           │                      │                  │
│  ┌────────▼────────┐  ┌─────────▼───────────────┐  │
│  │ StrategyBacktest│  │ BacktestEngine          │  │
│  │ (策略对比排名)   │  │ (滑动窗口回测)          │  │
│  └─────────────────┘  └─────────────────────────┘  │
│                                                     │
│  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ Analysis        │  │ FeatureEngineering      │  │
│  │ (综合分析报告)   │  │ (热冷号/遗漏/特征)     │  │
│  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────┐
│                    数据层                            │
│  ┌─────────────────────▼─────────────────────────┐  │
│  │  DataFetcher (src/data_fetcher.py)            │  │
│  │  500.com 趋势图解析 │ 增量合并 │ 试机号修复   │  │
│  └───────────────────────────────────────────────┘  │
│  data/ssq/  data/sd/  data/qlc/  data/kl8/  data/3d/│
└─────────────────────────────────────────────────────┘
```

---

## 项目结构

```
welfare_predict/
├── src/                          # 核心源码
│   ├── api.py                    # FastAPI Web 服务（/api/v1/ 路由前缀）
│   ├── session.py                # SQLite 持久化会话管理
│   ├── bootstrap.py              # 启动引导 + 数据同步
│   ├── scheduler.py              # APScheduler 定时任务（每日数据同步 + 智能训练）
│   ├── data_fetcher.py           # 数据抓取（KL8趋势图 + 增量合并）
│   ├── recommendation.py         # 多策略推荐引擎（四种策略）
│   ├── unified_pipeline.py       # ML 训练/预测管线（Stacking 实现）
│   ├── feature_engineering.py    # 特征工程（O(n²)→O(n) 优化）
│   ├── modeling.py               # XGBoost 基学习器
│   ├── model_lstm.py             # MLP/DNN 基学习器
│   ├── model_poisson.py          # 泊松先验基学习器
│   ├── model_stacking.py         # Stacking 元学习器
│   ├── model_io.py               # 统一模型序列化接口
│   ├── backtest.py               # 回测引擎
│   ├── strategy_backtest.py      # 策略回测对比
│   ├── analysis.py               # 综合分析报告
│   ├── user_history.py           # 用户历史记录
│   ├── config.py                 # 全局配置
│   └── common.py                 # 公共工具函数
├── static/                       # Web 前端
│   ├── index.html                # 主界面（多彩种切换 + ECharts）
│   └── login.html                # 登录页
├── scripts/                      # 命令行工具
│   ├── generate_recommendation.py
│   ├── strategy_ranking.py
│   ├── train.py
│   ├── predict.py
│   ├── get_data.py
│   ├── parse_lotterydata.py
│   ├── import_csv.py
│   ├── check_data.py             # 数据一致性校验
│   └── verify_data.py            # 数据验证
├── tests/                        # 测试套件（56 passed, 1 xfailed）
├── config/config.yaml            # 运行时配置
├── data/                         # 历史数据（git ignored）
│   ├── ssq/  sd/  qlc/  kl8/  3d/
│   └── users/                    # 用户数据
├── model/                        # 训练模型（git ignored）
├── docs/                         # 文档
├── Dockerfile                    # 端口 8080
├── docker-compose.yml
├── .env.example                  # 环境变量模板
└── start_server.bat              # Windows 启动脚本
```

---

## API 接口

所有业务接口前缀 `/api/v1/`，需登录后访问。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/login` | 登录 |
| POST | `/api/v1/logout` | 退出 |
| GET | `/api/v1/me` | 当前登录状态 |
| GET | `/health` | 健康检查（无需登录） |
| GET | `/api/v1/recommend/{code}` | 四种策略推荐 |
| POST | `/api/v1/predict/{code}` | ML 预测（`?method=xgb/lstm/poisson/stacking`） |
| GET | `/api/v1/history/{code}` | 历史数据（`?limit=30&offset=0`） |
| GET | `/api/v1/stats/{code}` | 统计图表数据（热冷号/遗漏） |
| GET | `/api/v1/ranking/{code}` | 策略排行榜（`?window=200&backtest=50`） |
| GET | `/api/v1/report/{code}` | 综合分析报告 |
| POST | `/api/v1/custom-recommend/{code}` | 自选AI推荐 |
| POST | `/api/v1/train/{code}` | 模型训练（`?method=xgb/poisson/stacking/lstm`） |
| POST | `/api/v1/train/{code}/all` | 批量训练所有可用方法 |
| GET | `/api/v1/train/methods` | 查询可用训练方法及状态 |
| GET | `/api/v1/train/{code}/status` | 训练状态查询（含 train_status.json） |
| POST | `/api/v1/data/update/{code}` | 增量数据更新 |

`{code}` 取值：`ssq`（双色球）、`sd`（福彩3D）、`qlc`（七乐彩）、`kl8`（快乐8）

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 异步 API 服务 |
| 前端 | 原生 HTML + ECharts | 零构建工具，多彩种切换 |
| ML 核心 | XGBoost + MLP + Poisson + Stacking | 四种基学习器 + 元学习器 |
| 特征工程 | NumPy/Pandas 向量化 | O(n²)→O(n) 优化 |
| 定时任务 | APScheduler BackgroundScheduler | 每日 01:03 BJT 数据同步 + 智能训练 |
| 会话存储 | SQLite | 持久化多用户会话 |
| 序列化 | ModelIO 统一接口 | joblib + TF native 自动分发 |
| 日志 | loguru + config.yaml | loguru 格式语法 |
| 部署 | Docker (Python 3.13-slim) / FastAPI Cloud | 端口 8080，含 healthcheck |

---

## 数据源

| 彩种 | 数据源 | 格式 | 同步方式 |
|------|--------|------|----------|
| 双色球 | datachart.500.com/ssq/ | 标准 history.shtml | 启动自动增量同步 |
| 福彩3D | datachart.500.com/3d/ | 标准 history.shtml | 启动自动增量同步 |
| 七乐彩 | datachart.500.com/qlc/ | 标准 history.shtml | 启动自动增量同步 |
| 快乐8 | datachart.500.com/kl8/ | 趋势图（80列遗漏值） | 数据源不可用 |

**天行数据源（已弃用）：**
- `TianyanAPISource` 已标记 deprecated，`fetch_history()` 将抛出 `ValueError`
- 天行平台（api.tianapi.com）230+ API 中无彩票类接口，原 `txapi/lottery/index` 返回 404
- 代码保留但不再使用，未来可替换为其他数据源

**KL8 趋势图解析要点：**
- 使用 `datachart.500.com/kl8/` 趋势图页面，非标准 `history.shtml`
- 80 列遗漏值格式，通过 `chartBall01` CSS 类识别当期开奖号码（20个）
- 日期从期号推导（"最新期≈当前日期"参考点推算）
- 启动时不自动同步（`ACTIVE_LOTTERY_CODES` 排除 kl8）
- 当前数据源不可用，前端已隐藏 KL8 标签页，后端逻辑保留待恢复

**福彩3D 试机号：**
- `_repair_sd_data(merge=True)` 从 3d/data.csv 合并试机号到 sd/data.csv
- 3d/data.csv 中 `tryCode=-1` 表示无数据，合并时自动过滤
- 前端历史数据表和开奖详情表均展示试机号列

---

## 定时任务与智能训练

### 数据同步触发方式

| 触发方式 | 时机 | 说明 |
|----------|------|------|
| 启动引导 | 服务启动时 | `sync_startup_data()` 检查数据充足性并增量同步 |
| 定时任务 | 每日 01:03 BJT | `APScheduler` CronTrigger 自动执行 |
| 手动触发 | 用户点击"更新数据" | `POST /api/v1/data/update/{code}` |

### 智能训练机制

- **train_status.json**：每个彩种 `data/{code}/train_status.json` 记录上次训练状态
  - `last_trained_issues`: 上次训练时的最新期号
  - `last_trained_at`: 上次训练时间
  - `trained_methods` / `failed_methods`: 成功/失败的方法列表
- **增量判断**：同步后比较当前最新期号与 `last_trained_issues`，仅在有新数据时触发训练
- **方法可用性检测**：`_get_unavailable_methods()` 运行时检测 TensorFlow 是否可用，不可用方法返回 400 错误并提示
- **批量训练**：`POST /train/{code}/all` 仅训练可用方法，跳过不可用方法

---

## 安全特性

- 密码通过环境变量 `LOTTERY_PASS` 注入，无默认值，空密码拒绝启动
- `require_auth()` 强制 401 认证校验
- SQLite 持久化会话，12小时过期，自动清理
- CORS 跨域限制
- 全局异常处理器（隐藏内部路径）
- 参数边界校验（limit, window, backtest）
- `DEBUG=false` 时启用 cookie Secure 标志

---

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `LOTTERY_PASS` | 是 | — | 登录密码（缺失时服务拒绝启动） |
| `LOTTERY_USER` | 否 | `admin` | 登录用户名 |
| `DEBUG` | 否 | `true` | 调试模式（生产环境设 false） |
| `PORT` | 否 | `8000` | 服务端口（DevCloud 必须 8080） |
| `CORS_ORIGINS` | 否 | `http://localhost:8000` | CORS 允许来源，逗号分隔 |
| `TIANYAN_API_KEY` | 否 | — | 天行数据 API key（已弃用，无需配置） |

---

## 命令行工具

```bash
# 生成推荐
python scripts/generate_recommendation.py

# 策略排行榜
python scripts/strategy_ranking.py --window 200 --backtest 50

# 训练模型（支持 xgb/lstm/poisson/stacking）
python scripts/train.py --name ssq --method xgb
python scripts/train.py --name ssq --method stacking

# 预测下一期
python scripts/predict.py --name ssq --method xgb

# 下载数据
python scripts/get_data.py --code ssq

# 数据一致性校验
python scripts/check_data.py
```

---

## 测试

```bash
# 全量测试
python -m pytest tests/ -v

# 单模块测试
python -m pytest tests/test_core.py -v      # 核心算法
python -m pytest tests/test_api.py -v       # API 端到端
python -m pytest tests/test_backtest.py -v  # 回测引擎
```

---

## 部署

### Docker

```bash
cp .env.example .env   # 编辑填写 LOTTERY_PASS
docker-compose up -d --build
# 访问 http://localhost:8080
```

### DevCloud（京东云）

- 端口必须为 **8080**（平台要求）
- `DEBUG=false` 以启用 cookie Secure 标志
- `CORS_ORIGINS` 设置为实际域名
- 支持 Dockerfile 部署（Python 3.13-slim 基础镜像）
- **必须**在平台控制台设置环境变量 `LOTTERY_PASS`（缺失时容器启动即崩溃）
- 平台会自动重写基础镜像为京东云镜像源、注入 APT/pip 国内镜像

### FastAPI Cloud

- 推送 GitHub 仓库后自动构建部署
- 平台运行 Python 3.13+，TensorFlow 不可用
- LSTM/Stacking 方法前端显示为"待升级"（disabled），XGBoost/Poisson 正常可用
- 端口由平台自动分配
- **必须**在平台控制台设置环境变量 `LOTTERY_PASS`
- 定时任务（APScheduler）在容器内正常运行

---

## 版本记录

### v2.5 (2026-08-08)

**云平台适配与自动化：**
- 新增 APScheduler 定时任务：每日 01:03（北京时间）自动增量同步 + 智能训练
- 智能训练机制：train_status.json 持久化，仅在有新数据时触发训练
- 批量训练接口：`POST /train/{code}/all` 一键训练所有可用方法
- 方法可用性查询：`GET /train/methods` 运行时检测 TensorFlow 可用性
- 不可用方法优雅处理：LSTM/Stacking 不可用时返回 400 + 原因说明，前端显示"待升级"

**数据源更新：**
- 天行数据源（TianyanAPISource）标记 deprecated，确认平台无彩票 API
- 快乐8数据源不可用，前端隐藏标签页，后端逻辑保留待恢复
- 福彩3D 历史数据表新增试机号列展示

**前端优化：**
- 方法选择器动态加载：不可用方法显示"(待升级)"并置灰禁选
- "更新数据"按钮改用批量训练接口，避免对不可用方法发起注定失败的请求
- 隐藏手动训练按钮（系统自动训练），保留更新数据按钮

**FastAPI Cloud 部署支持：**
- 推送 GitHub 自动构建部署
- Python 3.13+ 环境下 TensorFlow 不可用的完整降级方案

### v2.4 (2026-08-08)

**Python 3.13 升级：**
- Dockerfile 基础镜像从 `python:3.11-slim` 升级至 `python:3.13-slim`
- 依赖版本适配 Python 3.13：numpy>=2.1、scikit-learn>=1.6、pandas>=2.2.3、xgboost>=2.1、lxml>=5.2
- requirements-min.txt 标注 Python 3.13 兼容版本

**DevCloud 部署完善：**
- 明确 `LOTTERY_PASS` 环境变量为必填项，缺失时服务拒绝启动
- 补充平台控制台环境变量配置说明
- 补充平台镜像源自动重写说明

### v2.3 (2026-08-08)

**架构改进：**
- 快乐8恢复支持：实现趋势图页面解析器 `_parse_kl8_trend_chart()`
- 新增 API：`/ranking/{code}` 策略排行榜、`/report/{code}` 综合分析报告
- KL8 日期推导：使用"最新期≈当前日期"参考点推算

**健壮性改进：**
- KL8 启动时不自动同步，需手动触发
- KL8 趋势图解析增加严格校验：每期必须提取 20 个开奖号码
- 福彩3D 试机号修复：`_repair_sd_data()` 从 3d/data.csv 合并，过滤 tryCode=-1

**安全修复：**
- 移除硬编码密码，改为环境变量 `LOTTERY_PASS` 注入
- 空密码时服务拒绝启动（RuntimeError）
- 恢复 `require_auth()` 实际 401 认证校验

### v2.2 (2026-08-08)

**关键修复：**
- 前端玩法切换时页面加载卡死
- 七乐彩数据解析缺失蓝球_1和开奖日期字段
- session cookie Secure 标志导致 HTTP 下返回 401
- 前端 checkAuth() 异常导致整个页面 JS 崩溃

**架构改进：**
- 多彩种前端切换（双色球/福彩3D/七乐彩/快乐8 tab）
- red-special 球类型渲染（七乐彩 7红+1特别号）
- three 球类型渲染（福彩3D 百/十/个位）
- 启动时数据同步：自动检查数据充足性并增量同步
- API 路由统一迁移至 /api/v1/ 前缀

**DevCloud 部署适配：**
- Dockerfile/docker-compose.yml 端口改为 8080
- .env.example 包含 DevCloud 部署说明

---

*最后更新：2026-08-08 | 版本：v2.5*