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
# 编辑 .env 文件，修改 LOTTERY_PASS

# 3. 运行测试
python -m pytest tests/ -v

# 4. 启动服务
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
# 或使用启动脚本
start_server.bat

# 5. 浏览器访问
http://localhost:8000
# 默认账号: admin / （在 .env 中设置的密码）
```

服务启动时会自动检查数据充足性并同步增量数据（双色球、福彩3D、七乐彩）。

---

## Docker 部署

```powershell
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填写实际密码

# 2. 构建并启动
docker-compose up -d --build

# 3. 浏览器访问
http://localhost:8080
```

### DevCloud 部署

- 端口必须为 **8080**（平台要求）
- `DEBUG=false` 以启用 cookie Secure 标志
- `CORS_ORIGINS` 设置为实际域名

---

## 项目结构

```
welfare_predict/
├── src/
│   ├── api.py              # FastAPI Web 服务（v2.2: /api/v1/ 路由前缀）
│   ├── session.py          # SQLite 持久化会话管理
│   ├── bootstrap.py        # 启动引导 + 数据同步（v2.2 新增）
│   ├── model_io.py         # 统一模型序列化接口
│   ├── recommendation.py   # 多策略推荐引擎（四种策略）
│   ├── unified_pipeline.py # ML 训练/预测管线（含完整 Stacking 实现）
│   ├── feature_engineering.py  # 特征工程（已优化 O(n²)→O(n)）
│   ├── modeling.py         # XGBoost 基学习器
│   ├── model_lstm.py       # MLP/DNN 基学习器
│   ├── model_poisson.py    # 泊松先验基学习器
│   ├── model_stacking.py   # Stacking 元学习器
│   ├── backtest.py         # 回测引擎
│   ├── strategy_backtest.py    # 策略回测对比
│   ├── analysis.py         # 综合分析报告生成
│   ├── user_history.py     # 用户历史记录
│   ├── data_fetcher.py     # 数据抓取（v2.2: 增量合并模式）
│   ├── config.py           # 全局配置
│   └── common.py           # 公共工具函数
├── static/
│   ├── index.html          # Web 前端（v2.2: 多彩种切换 + 错误处理）
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
│   ├── conftest.py          # 测试配置
│   ├── test_core.py         # 核心算法测试
│   ├── test_preprocessing.py    # 预处理测试
│   ├── test_modeling.py         # 模型测试
│   ├── test_pipeline.py         # 管线测试
│   ├── test_config.py           # 配置测试
│   ├── test_api.py              # API 端到端测试
│   └── test_backtest.py         # 回测引擎测试
├── config/
│   └── config.yaml          # 运行时配置（loguru 格式）
├── data/                    # 历史数据（git ignored）
├── model/                   # 训练模型（git ignored）
├── users/                   # 用户数据（git ignored）
├── Dockerfile               # 镜像构建（端口 8080）
├── docker-compose.yml       # 编排文件（端口 8080）
├── .dockerignore            # 构建排除
├── .env.example             # 环境变量模板
├── .gitignore               # Git 忽略规则
├── requirements.txt         # Python 依赖
├── start_server.bat         # Windows 启动脚本
├── Makefile                 # 快捷命令
└── docs/
    ├── user_manual.md       # 用户手册
    ├── design.md            # 设计文档
    └── architecture.md      # 架构文档
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

| 代码 | 彩种 | 格式 | 状态 |
|------|------|------|------|
| `ssq` | 双色球 | 6红+1蓝 (1-33/1-16) | ✅ 完整支持 |
| `sd` | 福彩3D | 3位数字 (0-9) | ✅ 完整支持 |
| `qlc` | 七乐彩 | 7红+1特别号 (1-30) | ✅ 完整支持 |
| ~~`kl8`~~ | ~~快乐8~~ | ~~选10 (1-80)~~ | ❌ 已迁移至独立项目 |

---

## API 接口

所有业务接口前缀为 `/api/v1/`，需登录后访问。

```text
POST   /api/v1/login              登录
POST   /api/v1/logout             退出
GET    /api/v1/me                 当前登录状态
GET    /health                    健康检查

GET    /api/v1/recommend/{code}   四种策略推荐
POST   /api/v1/predict/{code}     ML 预测（?method=xgb/lstm/poisson/stacking）
GET    /api/v1/history/{code}     历史数据（?limit=30, ?offset=0）
GET    /api/v1/stats/{code}       统计图表数据（热冷号/遗漏）
POST   /api/v1/custom-recommend/{code}  自选AI推荐
POST   /api/v1/train/{code}       模型训练
GET    /api/v1/train/{code}/status  训练状态查询
POST   /api/v1/data/update/{code} 增量数据更新
```

### 安全特性

- ✅ CORS 跨域限制
- ✅ 全局异常处理器（隐藏内部路径）
- ✅ 参数边界校验（limit, window, backtest）
- ✅ SQLite 持久化会话（支持多用户）
- ✅ 硬编码密码已移除（必须通过 .env 注入）
- ✅ DEBUG 模式控制 cookie Secure 标志

---

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `LOTTERY_PASS` | ✅ | — | 登录密码 |
| `LOTTERY_USER` | ❌ | `admin` | 登录用户名 |
| `DEBUG` | ❌ | `true` | 调试模式（生产环境设为 false） |
| `PORT` | ❌ | `8000` | 服务端口（DevCloud 必须 8080） |
| `CORS_ORIGINS` | ❌ | `http://localhost:8000` | CORS 允许来源，逗号分隔 |
| `TIANYAN_API_KEY` | ❌ | — | 天行数据 API key |

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
```

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **Web 框架** | FastAPI + Uvicorn | 异步 API 服务 |
| **前端** | 原生 HTML + ECharts | 零构建工具，多彩种切换 |
| **ML 核心** | XGBoost + MLP(DNN) + Poisson + Stacking | 四种基学习器 + 元学习器 |
| **特征工程** | NumPy/Pandas 向量化 | 已优化性能热点 |
| **会话存储** | SQLite | 持久化多用户会话 |
| **序列化** | ModelIO 统一接口 | joblib + TF native 自动分发 |
| **部署** | Docker + Docker Compose | 端口 8080，含 healthcheck |
| **日志** | loguru + config.yaml | loguru 格式语法 |

---

## v2.2 更新记录（2026-08-08）

### 🔴 P0 — 关键修复
- 修复前端玩法切换时页面加载卡死（loadRecommend/loadHistory/loadStats 添加 try-catch）
- 修复七乐彩数据解析缺失蓝球_1和开奖日期字段（500.com 表格结构适配）
- 修复 session cookie Secure 标志导致 HTTP 下所有 API 返回 401（DEBUG 默认值改为 true）
- 修复前端 checkAuth() 异常导致整个页面 JS 崩溃

### 🟠 P1 — 架构改进
- 新增多彩种前端切换（双色球/福彩3D/七乐彩/快乐8 tab）
- 实现 red-special 球类型渲染（七乐彩 7红+1特别号）
- 实现 three 球类型渲染（福彩3D 百/十/个位）
- 新增启动时数据同步：自动检查数据充足性并增量同步
- 新增 download_history(merge=True) 增量合并模式，避免覆盖丢失历史数据
- API 路由统一迁移至 /api/v1/ 前缀

### 🟡 P2 — 健壮性改进
- 修复 config.yaml 日志格式为 loguru 语法（原 Python logging 格式导致乱码输出）
- bootstrap.py 增加格式防护：自动识别并忽略 Python logging 格式
- 快乐8 tab 改为可点击+迁移提示（移除灰色禁用外观）
- .env.example 新增 DEBUG/PORT/CORS_ORIGINS 配置项
- start_server.bat 全英文输出（修复 CMD GBK 编码问题）

### 🟢 P3 — DevCloud 部署适配
- Dockerfile/docker-compose.yml 端口改为 8080
- .env.example 包含 DevCloud 部署说明
- start_server.bat 自动激活 .venv、检查 .env、创建数据目录

---

## 免责声明

- 本系统**不保证中奖**
- 所有推荐基于历史数据统计，仅供娱乐参考
- 请理性购彩，量力而行
- 未成年人不得购买彩票

---

*最后更新：2026-08-08 | 版本：v2.2*