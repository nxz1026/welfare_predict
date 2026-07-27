# -*- coding: utf-8 -*-
"""
福彩推荐系统 — 架构设计文档（v2.1 更新版）。

P4-04: 同步更新架构图以反映当前实际代码状态。
"""

# 福彩推荐系统 — 架构设计文档

## 一、系统概述

### 1.1 产品定位
面向彩票店老板的号码推荐工具。**不是预测中大奖的机器**，而是用历史统计数据给顾客提供"有理论依据"的推荐。

### 1.2 核心价值
- **专业形象**：用数据说话，区别于"随便选几个"
- **四策略覆盖**：保守/激进/平衡/玄学，满足不同顾客心理
- **反赌徒谬误**：明确告知"不保证中奖"，提供策略 vs 随机对比

### 1.3 技术选型

| 层级 | 技术 | 选择理由 |
|------|------|----------|
| Web 框架 | **FastAPI** | 异步高性能，自动 OpenAPI 文档 |
| 前端 | **HTML + ECharts** | 零构建工具，单文件即可运行 |
| ML 核心 | **XGBoost + MLP + Poisson + Stacking** | 多模型集成，降低单一模型偏差 |
| 特征工程 | **NumPy / Pandas** | 高效数值计算，已做向量化优化 |
| 数据源 | **500.com datachart** | 权威公开数据，稳定可靠 |
| 会话存储 | **SQLite** | 轻量持久化，支持多用户（v2.1 新增） |
| 序列化 | **ModelIO 统一接口** | joblib + TF native 自动分发（v2.1 新增） |
| 日志 | **loguru** | 结构化日志，支持 config.yaml 配置（v2.1 修复） |
| 部署 | **Docker + Compose** | 一键启动，含 healthcheck |

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                      用户界面层                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  index.html  │  │  login.html  │  │  REST API    │  │
│  │  (ECharts)   │  │  (表单认证)   │  │  (FastAPI)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
└─────────┼─────────────────┼──────────────────┼──────────┘
          │                 │                  │
┌─────────┼─────────────────┼──────────────────┼──────────┐
│         ▼                 ▼                  ▼          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              API 服务层 (src/api.py)            │    │
│  │  ┌─────────────┐  ┌──────────────────────────┐  │    │
│  │  │ CORS 中间件  │  │ 全局异常处理器(隐藏路径) │  │    │
│  │  │ 参数边界校验 │  │ /health 健康检查         │  │    │
│  │  └─────────────┘  └──────────────────────────┘  │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                               │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │           会话管理层 (src/session.py)            │    │
│  │     SQLite 持久化存储 / 多用户支持 / 过期清理    │    │
│  └──────────────────────┬──────────────────────────┘    │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────┐
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │              业务逻辑层                           │    │
│  │                                                  │    │
│  │  ┌────────────────┐  ┌──────────────────────┐   │    │
│  │  │ Recommendation │  │ UnifiedPipeline      │   │    │
│  │  │ Engine         │  │ (训练/预测管线)       │   │    │
│  │  │ (四策略推荐)    │  │                      │   │    │
│  │  └───────┬────────┘  └──────┬───────────────┘   │    │
│  │          │                   │                   │    │
│  │  ┌───────▼────────┐  ┌──────▼───────────────┐   │    │
│  │  │ StrategyBacktest│  │ BacktestEngine       │   │    │
│  │  │ (策略对比排名)  │  │ (滑动窗口回测)       │   │    │
│  │  └────────────────┘  └──────────────────────┘   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────┐
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │              ML 模型层                            │    │
│  │                                                  │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │    │
│  │  │ XGBoost  │ │ MLP/DNN  │ │ Poisson  │        │    │
│  │  │ Predictor│ │ Predictor│ │ Prior    │        │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘        │    │
│  │       │            │            │               │    │
│  │  ┌────▼────────────▼────────────▼──────────┐   │    │
│  │  │     StackingEnsemble (元学习器)          │   │    │
│  │  │     LogisticRegression meta-learner     │   │    │
│  │  └─────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────┐
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │           特征工程层 (feature_engineering.py)    │    │
│  │                                                  │    │
│  │  热冷号(滑动窗口O(n)) │ 间隔(skip/interval)      │    │
│  │  和值/跨度 │ 奇偶比 │ 质合比(向量化) │ AC值      │    │
│  │  历史统计(均值/标准差)                             │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────┼───────────────────────────────┐
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │           数据层 (data_fetcher.py)               │    │
│  │                                                  │    │
│  │  500.com datachart → BeautifulSoup 解析 → CSV   │    │
│  │  支持彩种: ssq/dlt/sd/qlc/pls/qxc               │    │
│  │  (KL8 已迁移至独立项目)                           │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
500.com API
    │
    ▼
[data_fetcher.py] ──→ data/<code>/data.csv
    │                     │
    ▼                     ▼
[feature_engineering.py] → 特征矩阵 (N × D)
    │
    ├─→ [XGBoostPredictor] ──┐
    ├─→ [MLPPredictor]      ──┤
    ├─→ [PoissonPrior]      ──┤──→ [StackingEnsemble]
    │                          │     (meta-learner)
    ▼                          ▼
[UnifiedPipeline.train()]  [UnifiedPipeline.predict()]
    │                          │
    ▼                          ▼
model/<code>/method/model.pkl → 推荐号码
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
           [Recommendation]  [BacktestEngine]  [Analysis]
           (四策略推荐)       (ROI/命中率)     (综合报告)
                    │             │             │
                    ▼             ▼             ▼
              ──→ [api.py] ──→ JSON API ──→ 前端 ECharts
```

---

## 三、模块说明

### 3.1 核心模块

| 模块 | 职责 | 关键类/函数 |
|------|------|-------------|
| `api.py` | Web 服务入口 | FastAPI app, 登录/授权/业务接口 |
| `session.py` | 会话管理 | create_session, validate_session, cleanup_expired |
| `recommendation.py` | 多策略推荐 | ConservativeStrategy, AggressiveStrategy, BalancedStrategy, MysticStrategy |
| `unified_pipeline.py` | ML 管线 | UnifiedPipeline, StackingEnsemble |
| `feature_engineering.py` | 特征工程 | build_feature_matrix (110维), compute_hot_cold_features (已优化) |
| `modeling.py` | XGBoost 模型 | XGBoostPredictor |
| `model_lstm.py` | MLP/DNN 模型 | LSTMPredictor (内部为 MLP 架构) |
| `model_poisson.py` | 泊松先验 | PoissonPrior |
| `backtest.py` | 回测引擎 | BacktestEngine, calculate_prize, PRIZE_MONEY 常量 |
| `strategy_backtest.py` | 策略对比 | StrategyBacktestEngine, StrategyPerformance |
| `analysis.py` | 分析报告 | generate_comprehensive_report |
| `config.py` | 配置管理 | LotteryModelConfig, LOTTERY_CONFIGS |
| `data_fetcher.py` | 数据抓取 | download_history, load_history |
| `model_io.py` | 序列化统一 | ModelIO.save/load (P1-04 新增) |
| `bootstrap.py` | 启动引导 | configure_logging (P3-02 新增) |

### 3.2 新增/重构模块（v2.1）

| 模块 | 变更类型 | 说明 |
|------|----------|------|
| `session.py` | **新增** | SQLite 持久化会话，替代内存单用户模式 |
| `model_io.py` | **新增** | 统一 save/load 接口，自动分发 joblib/TF |
| `bootstrap.py` | **新增** | loguru 读取 config.yaml 日志配置 |
| `tests/test_api.py` | **新增** | API 端到端测试（登录/授权/参数校验） |
| `tests/test_backtest.py` | **新增** | 回测引擎单元测试 |
| `.env.example` | **新增** | 环境变量模板 |
| `.dockerignore` | **新增** | Docker 构建排除规则 |
| `api.py` | **重构** | SQLite 会话 + CORS + 异常处理 + 健康检查 |
| `unified_pipeline.py` | **重构** | 完整 Stacking 实现（不再静默降级） |
| `model_lstm.py` | **重构** | LSTM → MLP/DNN（修正时序维度 bug） |
| `feature_engineering.py` | **优化** | 滑动窗口 Counter + 向量化质合计算 |
| `recommendation.py` | **优化** | 向量化蓝球选择 + 类型注解修复 |
| `backtest.py` | **优化** | 魔法数字提取为 PRIZE_MONEY 常量 |
| `config.py` | **清理** | 移除 KL8 LotteryModelConfig |
| `data_fetcher.py` | **清理** | 移除 KL8 分支逻辑 |
| `conftest.py` | **修复** | name_path → PATHS 修复 |
| `Dockerfile` | **安全** | ARG 注入替代 ENV 硬编码 |
| `requirements.txt` | **修复** | 补齐 xgboost + joblib |

---

## 四、安全设计（v2.1 新增）

### 4.1 认证与授权
- Cookie-based Session（HttpOnly, SameSite=Lax）
- SQLite 持久化存储，支持多终端同时登录
- 7 天默认过期时间，启动时自动清理过期会话

### 4.2 输入验证
- 所有 Query 参数使用 Pydantic `Query(ge=..., le=...)` 边界约束
- 全局异常处理器捕获未处理异常，仅返回安全信息
- 彩种代码白名单校验（LOTTERY_CONFIGS 字典查找）

### 4.3 部署安全
- 密码通过 `--build-arg` 或 `.env` 文件注入，禁止硬编码
- LOTTERY_SECRET 用于未来 cookie 签名扩展
- `.env` 在 `.gitignore` 中，`.env.example` 提供模板

---

## 五、性能优化（v2.1）

| 优化项 | 原复杂度 | 优化后 | 影响 |
|--------|----------|--------|------|
| 热冷号计算 | O(n × window × num_classes) | O(n × sequence_len) | 3000期 < 100ms |
| 蓝球遗漏扫描 | O(n × blue_range) | O(n) (value_counts) | < 1ms |
| 质合特征计算 | O(n × sequence_len) 循环 | NumPy 向量化 | < 10ms |
| Docker 镜像大小 | 包含全部文件 | .docker排除 | 减少 ~40% |

---

## 六、已知限制与未来规划

### 6.1 当前限制
- 单机部署（无负载均衡）
- 用户历史记录仍为文件存储（待迁移 SQLite）
- 无定时数据更新机制（需手动或 cron）
- 前端为单文件 HTML（~400 行）

### 6.2 未来规划（P5 Roadmap）
- P5-01: 用户历史记录 SQLite 化
- P5-02: 定时数据更新（cron/GitHub Actions）
- P5-03: CI/CD Pipeline（lint → test → build → push）
- P5-04: API 版本化（/api/v1/...）
- P5-05: 反赌徒谬误可视化页面
- P5-06: WebSocket 实时推送
- P5-07: 前端现代化（Vue/React SPA）

---

*文档版本: v2.1 | 最后更新: 2026-07-27 | 基于 welfare_predict 项目 v2.1 代码*
