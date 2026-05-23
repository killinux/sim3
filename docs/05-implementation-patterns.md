# 实现架构与技术栈

## 一、Agent 架构设计

### 1.1 用户状态表示（三层结构）

基于 Agent4Rec、A/B Agent (KDD 2026)、SimTok、PersonaAct、SimUSER 的架构分析。

#### Layer 1: 静态人口统计

```python
class Demographics:
    user_id: str
    age_bucket: str        # "18-24", "25-30", ...
    gender: str            # "male", "female", "unknown"
    city_tier: int         # 1-5 (一线到五线)
    device_type: str       # "ios_high", "android_mid", ...
    personality: dict      # Big Five 特征分值
```

SimTok 在抖音类模拟中正好使用了这些特征。

#### Layer 2: 可演化偏好

```python
class Preferences:
    interest_vector: dict[str, float]   # 品类 -> 权重
    favorite_creators: list[str]         # 关注的创作者 ID
    content_style_pref: dict             # 偏好的内容风格
    novelty_seeking: float               # 猎奇倾向 [0, 1]
    trend_sensitivity: float             # 趋势敏感度 [0, 1]
```

偏好演化方法：
- **快速更新**: 每次交互后对 interest_vector 做指数移动平均
- **慢速反思**: 每 N 个 session 由 LLM 做"反思"，更新兴趣摘要

#### Layer 3: 行为签名（从真实数据校准）

```python
class BehavioralSignature:
    avg_session_duration: float    # 平均 session 时长（分钟）
    sessions_per_day: float        # 日均 session 数
    videos_per_session: float      # 每 session 视频数
    engagement_rates: dict         # 各动作类型的参与率
    peak_hours: list[float]        # 活跃时段概率分布（24维）
    swipe_profile: str             # "fast_scanner" / "deliberate_watcher" / "binge_viewer"
```

### 1.2 动作空间设计

#### 离散动作

```python
class UserAction(Enum):
    SWIPE_NEXT = "swipe_next"           # 快速滑过
    WATCH_PARTIAL = "watch_partial"      # 部分观看
    WATCH_FULL = "watch_full"            # 完整观看
    WATCH_REPLAY = "watch_replay"        # 重播
    LIKE = "like"
    COMMENT = "comment"
    SHARE = "share"
    SAVE = "save"
    FOLLOW_CREATOR = "follow_creator"
    VISIT_PROFILE = "visit_profile"
    CLICK_HASHTAG = "click_hashtag"
    SEARCH = "search"
    END_SESSION = "end_session"
```

#### 连续参数

```python
class ActionParams:
    watch_duration_ratio: float   # 0.0 ~ 2.0+（>1 表示重播）
    comment_text: str | None      # 评论内容（可选）
    decision_latency: float       # 决策延迟（秒）
```

#### 观看时长建模

观看时长分为四个组分的混合分布，兴趣度和疲劳度调节混合权重：

| 组分 | 分布 | 含义 |
|------|------|------|
| 即跳 | Exp(λ=0.7), 截断在 ~1.5s | 不感兴趣，立即跳过 |
| 部分观看 | Beta(2,3) × duration | 有一定兴趣但未看完 |
| 完整观看 | Beta(8,2) × duration | 高兴趣，看完 |
| 重播 | Beta(8,2) × duration × (1+replay) | 非常感兴趣，重播 |

#### 多步动作链的条件概率

动作之间有条件依赖：高完播后点赞的 base rate 约 15%，但：
- P(评论 | 已点赞) = base_rate × 3
- P(分享 | 已点赞) = base_rate × 2.5
- P(关注 | 已点赞+评论) = base_rate × 4

形成自然的动作链：观看 -> 点赞 -> 评论 -> 分享，无需每个子动作单独调用 LLM。

### 1.3 记忆与上下文管理

#### 三层记忆架构

| 层级 | 名称 | 存储 | Token 预算 | 更新频率 |
|------|------|------|-----------|---------|
| Tier 1 | 工作记忆 | 最近 ~30 次交互，纯文本 | ~500 | 每步 |
| Tier 2 | 情节记忆 | 关键事件（评论/分享/关注/高参与）KV 存储 | 按需检索 ~200 | 显著交互时 |
| Tier 3 | 语义记忆 | LLM 生成的压缩摘要（session/日/周） | ~300 | 定期 |

**每 agent 每步 Token 预算**:
- Profile: ~1500 tokens
- 记忆上下文: ~500 tokens
- 当前视频 + 提示词: ~300 tokens
- **输入合计: ~2300 tokens**
- 输出: ~200 tokens
- 1000 agent 每步: ~2.5M input + ~200K output tokens

---

## 二、模拟引擎设计

### 2.1 并发 Agent 执行

#### 推荐架构: 异步编排器 + 批量 LLM 调度

```
┌─────────────────────────────────────────┐
│           Event-Driven Simulator         │
│  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │ Session  │  │ Content  │  │ Social │ │
│  │Generator │  │  Pool    │  │ Graph  │ │
│  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       └──────────────┼────────────┘      │
│                      ▼                   │
│           ┌──────────────────┐           │
│           │  Agent Scheduler  │           │
│           │  (async batching) │           │
│           └────────┬─────────┘           │
│                    ▼                     │
│           ┌──────────────────┐           │
│           │   LLM Endpoint   │           │
│           │ (vLLM / API)     │           │
│           └────────┬─────────┘           │
│                    ▼                     │
│           ┌──────────────────┐           │
│           │ Metrics Collector │           │
│           └──────────────────┘           │
└─────────────────────────────────────────┘
```

**调度器核心逻辑**:
- 收集待处理的 agent 请求到批次（最大 32-64 个/批，50ms 超时收集部分批次）
- 通过 aiohttp 发送到 vLLM endpoint
- 分发响应回等待中的 agent coroutine
- 信号量限制最大并发请求（64-128）

#### 模型路由优化

```
简单决策（~80% 动作）→ Qwen2.5-7B
  - SWIPE_NEXT, WATCH_PARTIAL 等低复杂度动作

复杂决策（~20% 动作）→ Qwen2.5-72B
  - COMMENT（需生成文本）
  - 高参与后的动作链决策
  - 冷启动/新场景决策
```

成本降低 **60-70%**。

### 2.2 推荐系统集成

#### 三个选项

| 选项 | 方法 | 复杂度 | 真实度 |
|------|------|--------|--------|
| **A: 代理模型（推荐）** | 训练轻量 two-tower / SASRec | 中 | 高 |
| B: 规则引擎 | 40%兴趣+20%关注+20%热搜+20%探索 | 低 | 中 |
| C: RecoWorld | 推荐器+用户多轮交互+反思 | 高 | 最高 |

**选项 A 详细设计**:
- 在历史交互数据上训练 SASRec 或 two-tower 模型
- 对候选内容打分（用户 embedding × 内容 embedding）
- 加入 ε-greedy 探索（ε 由用户 novelty_seeking 调节）
- 关键要求：支持 AB 变体切换（不同排序参数 per variant）

**内容池**:
- 从真实平台数据加载（5万视频元数据 + embedding）
- 或用真实分布生成合成内容池

### 2.3 时间与 Session 模拟

#### 推荐: 事件驱动模拟 + 时钟同步

```python
# 核心事件类型
SessionStartEvent(agent_id, timestamp)
VideoDecisionEvent(agent_id, video_id, timestamp)
SessionEndEvent(agent_id, timestamp)
DailyReflectionEvent(agent_id, timestamp)
```

**优先队列** (heapq) 管理事件。每模拟天开始时：
1. 为每个 agent 生成 session-start 事件（Poisson 分布的 session 数）
2. Session 开始时间从小时活跃度分布中采样
3. Session 内动态生成 video-decision 和 session-end 事件

**时间压缩**:
- 事件驱动无空闲周期
- 1 模拟天实际处理事件量：
  - 每 agent 每 session: 10-50 个视频决策
  - 每 agent 每天: 2-5 个 session
  - 1000 agent: **20,000 - 250,000 事件/模拟天**

**时间模式建模**:

| 模式 | 参数 |
|------|------|
| 小时分布 | 峰值 19:00-22:00，24 维概率分布 |
| 周末效应 | 周末 +15-18% |
| 节假日 | 春节 +50%，暑假 +20% |

### 2.4 疲劳系统（来自 A/B Agent）

```
fatigue += (1 - satisfaction) × fatigue_rate
P(end_session) = α × fatigue² + β × fatigue
```

- fatigue 随每个视频观看增加，满意度高时增长慢
- Session 结束概率是 fatigue 的二次函数（慢启动 -> 快速结束）
- 防止 agent 不切实际地"无限浏览"

---

## 三、验证 Pipeline 设计

### 3.1 指标收集

**核心指标**:
- 平均观看时长、完播率
- 每 session 视频数、session 时长
- 点赞/评论/分享/关注/收藏率
- 次日留存率、7 日留存率
- 满意度（agent 自报告）
- 内容多样性（品类信息熵）

### 3.2 Treatment Effect 计算

加权 treatment effect:
```
τ̂ = Σ w_i × Y_i(treatment) / Σ w_i(treatment)
   - Σ w_i × Y_i(control) / Σ w_i(control)
```

其中 w_i 是后分层权重。

#### 统计检验

- 500 control + 500 treatment 可检测 Cohen's d >= 0.18（约 3-5% 相对效果）
- **推荐 Bootstrap 置换检验**（优于 t 检验，因为参与行为非正态分布）
- **贝叶斯方法**: 计算 P(treatment > control)，小样本更可解释
- **多重比较修正**: Benjamini-Hochberg (FDR 控制) 优于 Bonferroni
  - BH 控制 FDR 低于 alpha 同时保持更高检验力
  - 同时检验多个参与指标时尤其重要
- **增加检验力**: 多模拟天聚合 —— 每天提供每 agent 的独立观测

### 3.3 历史 AB 测试回放

**流程**:
1. 按原始实验比例分配 agent 到各 variant
2. 用实际 variant 参数配置代理推荐系统
3. 运行与原实验相同时长的模拟
4. 比较模拟 treatment effect 与真实结果

**关键准确度指标**: "方向准确率" —— 多少比例的指标方向变化（正/负 lift）被正确预测。

**目标**: 在 3-5 个回放实验中达到 **>80% 方向准确率**后，才信任系统做新实验预验证。

**需要的历史数据**:
- Variant 配置参数
- 真实结果（control 均值、treatment 均值、lift、p 值）
- 用户分层特征
- （可选）交互日志样本

---

## 四、技术栈推荐

### 核心组件

| 组件 | 推荐方案 | 备选 | 理由 |
|------|---------|------|------|
| **Agent 编排** | AgentScope（原型）/ 自定义 async（生产） | LangGraph | AgentScope 在 1万+ agent 上验证过；自定义给最大控制 |
| **LLM 推理** | vLLM | SGLang | 吞吐量比 TGI 高 3.67x；TGI 已于 2025.12 进入维护模式 |
| **LLM 模型** | Qwen2.5-72B-Instruct (AWQ量化) | GPT-4o-mini (API) | 自部署成本可预测；Qwen 与阿里生态契合 |
| **模拟引擎** | 自定义事件驱动 (heapq+asyncio) | Mesa 3 | Mesa/SimPy 不适合顺序用户-内容交互 |
| **数据处理** | DuckDB + Polars | Pandas | DuckDB 做日志 SQL 分析；Polars 做快速 DataFrame 变换 |
| **因果推断** | DoWhy (PyWhy) | EconML | DoWhy 提供 identify-estimate-refute pipeline |
| **统计检验** | scipy + statsmodels | - | Bootstrap、置换检验、BH 修正 |
| **仪表盘** | Streamlit + Plotly | Gradio | Streamlit 更适合数据密集型仪表盘 |

### 自部署推理硬件

| 配置 | GPU | 每步时间 | 适用 |
|------|-----|---------|------|
| 最低 | 2x A100 80GB | ~14 分/步 | 原型验证 |
| 推荐 | 4-8x A100 80GB | ~3-7 分/步 | 生产使用 |
| 模型路由 (7B+72B) | 2x A100 | ~3 分/步 | 性价比最优 |

### 框架对比

| 框架 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **AgentScope** (阿里) | 20分钟快速开始，原生 DashScope/Qwen 集成，1万+ agent 验证 | 偏通用，需定制推荐逻辑 | 快速原型 |
| **OASIS** (CAMEL-AI) | 已有社交媒体模型+推荐+23动作，Apache 2.0 | SQLite 瓶颈，文本导向非视频 | 社交媒体模拟基座 |
| **LangGraph** | 复杂 agent workflow | 非大规模同质 agent 模拟优化 | 单个复杂 agent |
| **Mesa 3** | 学术标准 ABM 框架 | 规则为主，无 LLM 集成 | 宏观层参考 |
| **AgentTorch** (MIT) | 可微分，GPU 原生，840万 agent | AGPL-3.0，需从零构建推荐场景 | 大规模可微分模拟 |

---

## 五、项目结构建议

```
sim3/
├── docs/                          # 文档
├── src/
│   ├── agents/
│   │   ├── user_agent.py          # 用户 agent 定义
│   │   ├── creator_agent.py       # 创作者 agent（方案C）
│   │   ├── persona.py             # Persona 数据结构
│   │   └── memory.py              # 三层记忆管理
│   ├── simulation/
│   │   ├── engine.py              # 事件驱动模拟引擎
│   │   ├── scheduler.py           # 异步 LLM 批量调度
│   │   ├── session.py             # Session 生成与疲劳模型
│   │   └── time_model.py          # 时间模式建模
│   ├── recommendation/
│   │   ├── surrogate_recsys.py    # 代理推荐系统
│   │   ├── content_pool.py        # 内容池管理
│   │   └── ab_variant.py          # AB 变体配置
│   ├── scaling/
│   │   ├── clustering.py          # 用户聚类/原型发现
│   │   ├── weighting.py           # Raking + 重要性加权
│   │   └── interpolation.py       # 统计插值到 10 亿级
│   ├── validation/
│   │   ├── distribution_metrics.py  # KL/JS/Wasserstein/MMD
│   │   ├── ab_replay.py           # 历史 AB 回放
│   │   ├── aa_test.py             # A/A 验证
│   │   └── calibration.py         # 因果后校准
│   ├── analysis/
│   │   ├── treatment_effect.py    # Treatment effect 计算
│   │   ├── statistical_tests.py   # Bootstrap/贝叶斯检验
│   │   └── dashboard.py           # Streamlit 仪表盘
│   └── data/
│       ├── loaders/               # 数据加载器（KuaiRand等）
│       └── preprocessing/         # 数据预处理
├── configs/
│   ├── personas/                  # 1000 个 persona 定义
│   ├── simulation.yaml            # 模拟配置
│   └── experiment.yaml            # AB 实验配置
├── scripts/
│   ├── run_simulation.py          # 运行模拟
│   ├── run_aa_test.py             # 运行 A/A 验证
│   ├── run_ab_replay.py           # 运行 AB 回放
│   └── generate_personas.py       # 生成 personas
└── tests/
```

---

## 六、实施路线图

| 阶段 | 时间 | 目标 | 交付 |
|------|------|------|------|
| Phase 0 | 第1-3周 | 基础搭建 | vLLM 部署, agent 数据结构, 事件驱动引擎骨架, 合成内容池 |
| Phase 1 | 第4-6周 | 核心模拟 | 异步调度器, 代理推荐系统+AB变体, Session 生成, 100 agent 验证 |
| Phase 2 | 第7-9周 | 规模化 | 扩展到 1000 agent, 模型路由, 记忆压缩, 语义缓存, prompt 蒸馏 |
| Phase 3 | 第10-12周 | 验证 | 指标 pipeline (DuckDB/Polars), 3-5 个历史实验回放, 迭代至方向准确率 >80% |
| Phase 4 | 第13-15周 | 生产化 | Streamlit 仪表盘, DoWhy 因果推断, 文档 |
