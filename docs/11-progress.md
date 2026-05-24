# 项目进度记录

## 一、系统原理

### 1.1 核心思路

用 LLM 扮演用户，让它"看"推荐的视频，做出和真人一样的决策（看/跳/点赞/评论），然后统计这些模拟行为的分布，来预测真实 AB 实验的效果。

```
真实世界                              模拟世界
─────────                            ─────────
10亿用户 → 推荐系统 → 看视频           1000个LLM Agent → 代理推荐 → "看"视频
         → 点赞/跳过/评论                              → 输出JSON决策
         → 收集指标                                    → 收集指标
         → AB实验结论                                  → 预测AB结论
```

### 1.2 五层架构

| 层级 | 功能 | 实现 |
|------|------|------|
| **Persona 层** | 定义每个 Agent 的身份 | 人口统计 + 兴趣向量 + 行为签名 + 社交特征（活跃度/从众度/多样性） |
| **推荐层** | Agent 看到什么 | Two-tower 代理推荐系统，基于 embedding 相似度 + epsilon-greedy 探索 |
| **决策层** | Agent 怎么反应 | LLM 输出 interest_level → 分位数映射校准 → Beta 分布采样 → watch_ratio |
| **记忆层** | Agent 状态演化 | 三层记忆（工作记忆30条 + 情节记忆 + 语义摘要）+ 兴趣更新 + 疲劳累积 |
| **分析层** | 汇总结果 | 分 variant 指标计算 + Bootstrap 置信区间 + BH 多重比较修正 + A/A 验证 |

### 1.3 单次决策流程

```
1. 构建 Prompt
   System: "你是25岁男性，喜欢游戏，活跃度高..."
   User:   "下一个视频：[游戏] 30s | 5000 likes
            你最近看了: ... (最近10条)
            疲劳度: 0.35
            请用JSON回复 interest_level (1-10)"

2. LLM 返回
   {"decision":"watch_full", "interest_level":8, "continue":true}

3. 分位数映射校准
   raw interest=8 → 经验CDF百分位 → 目标CDF反查 → calibrated interest

4. Beta 分布采样
   interest 8-9 → watch_ratio ~ Beta(4, 1.9) × 0.35 + 0.51
   参与动作: like/comment/share 由 persona base_rate × boost 独立采样

5. 更新状态
   记忆 += 这次交互
   兴趣向量微调（对游戏 +0.05）
   疲劳 += (1-满意度) × 疲劳率
   判断是否退出 session
```

### 1.4 1K Agent → 1B 用户的缩放原理

来自统计学的**分层抽样**：

1. 将 10 亿用户按行为特征聚类为 1000 个原型
2. 每个 Agent 代表一个原型，携带人口权重（如：200万人）
3. 加权聚合：`模拟指标 = Σ(agent_i指标 × 权重_i) / Σ权重`

跟民意调查用 1000 人代表 3 亿人口是同一个原理。

### 1.5 AB 实验验证原理

```
1000 agent ──┬── 500 treatment（如：高多样性推荐）
             └── 500 control  （如：低多样性推荐）

两组唯一区别 = 推荐算法参数不同
差异 = treatment effect 估计
```

### 1.6 四层验证体系

| 层级 | 问什么 | 怎么测 | 当前状态 |
|------|--------|--------|---------|
| A/A 验证 | 系统有没有偏差？ | 两组相同，效果应为 0 | ✅ PASS |
| 分布对比 | 分布像不像真实？ | Wasserstein / JS 散度 | ✅ 基本对齐 |
| 方向一致 | 方向对不对？ | 回放历史 AB | ❌ 待做 |
| 幅度准确 | 幅度准不准？ | Pearson 相关 | ❌ 待做 |

---

## 二、当前进展

### 2.1 已完成的调研（docs/01-10）

- 30+ 论文、10+ 开源项目调研
- 14 种可能方案穷举与对比
- OASIS / AgentTorch / Agent4Rec 三个框架深入代码走读
- 推荐方案 A（混合多层+校准）→ B（数字孪生+可微分）→ C（生态协同）的渐进路径

### 2.2 已实现的原型代码

| 模块 | 文件 | 行数 | 状态 |
|------|------|------|------|
| Persona | `src/agents/persona.py` | 170 | ✅ |
| User Agent | `src/agents/user_agent.py` | ~250 | ✅ 含分位数映射 |
| Memory | `src/agents/memory.py` | 110 | ✅ |
| Content Pool | `src/recommendation/content_pool.py` | 130 | ✅ |
| Surrogate RecSys | `src/recommendation/surrogate_recsys.py` | 100 | ✅ |
| Simulation Engine | `src/simulation/engine.py` | 195 | ✅ |
| LLM Scheduler | `src/simulation/scheduler.py` | 130 | ✅ |
| Metrics | `src/analysis/metrics.py` | 70 | ✅ |
| Treatment Effect | `src/analysis/treatment_effect.py` | 120 | ✅ |
| Distribution Metrics | `src/validation/distribution_metrics.py` | 90 | ✅ |
| A/A Test | `src/validation/aa_test.py` | 70 | ✅ |
| Run Script | `scripts/run_simulation.py` | 150 | ✅ |

### 2.3 运行验证结果

#### KuaiRand 真实数据验证 — 当前最优（v12f, 50 agent）

| 指标 | 快手真实 | 模拟 v12f (5次均值) | 标准差 | 状态 |
|------|---------|-------------------|--------|------|
| skip_rate | 47.98% | **48.3%** | ±2.2pp | ✅ |
| completion_rate | 7.78% | **8.69%** | ±1.2pp | ✅ |
| avg_watch_ratio | 0.445 | **0.414** | ±0.015 | ⚠️ 差7% |
| like_rate | 0.48% | **0.11%** | ±0.15pp | ⚠️ 偏低 |
| comment_rate | 0.03% | 0% | - | ⚠️ 样本少 |
| share_rate | 0.03% | ~0.03% | - | ✅ |

#### 校准迭代全历程

| 版本 | 核心改动 | skip_rate (std) | completion | avg_wr | 判断 |
|------|---------|----------------|------------|--------|------|
| 原始 | 无校准 | 38% | 50% | 0.72 | LLM 太友好 |
| v1-v5 | prompt/后处理校准 | 33-84% | 0-50% | 0.17-0.62 | 不稳定 |
| v6 | 混合模型(interest+Beta) | 56% | 20.8% | 0.393 | 架构确定 |
| v7-v11 | Beta参数调优 | 43-59% (19.5pp) | **7.32%** | 0.376 | completion好, 方差大 |
| **v12f** | **+分位数映射+CDF调优** | **48.3% (2.2pp)** | **8.69%** | **0.414** | **✅ 当前版本** |

---

## 三、已修正的问题

### 3.1 Like Rate 过高（52% → 0.11%）

**问题**: 首次运行 like_rate 达 52%，真实平台约 0.48%。

**根因**: LLM 的 RLHF 训练使其倾向"友好"和"积极"（正向偏差）。

**修正**: 经 v1-v6 六轮迭代，最终采用**混合模型**——LLM 只输出 interest_level (1-10)，参与动作完全由统计模型生成。

### 3.2 Completion Rate 过高（20.8% → 8.69%）

**问题**: v6 的 completion_rate 20.8%，真实 7.78%。

**根因**: Band 4 `Beta(6,1.5)*0.2+0.8` 最小值 0.8，interest=10 的视频 100% 完播。

**修正**: 经 v7-v11 六轮 Beta 参数迭代，调整为 `Beta(5,2)*0.25+0.68`（范围[0.68,0.93]），详见 [13-beta-calibration.md](13-beta-calibration.md)。

### 3.3 LLM 方差过大（skip std 19.5pp → 2.2pp）

**问题**: 50 agent 跑 5 次，skip_rate 从 19% 到 73%。方差来自 LLM interest_level 输出的 batch-level correlation。

**尝试失败的方案**:
- affinity prior blending：先验太低，砍掉了高 interest 尾部
- 均值修正：无法处理双峰分布（LLM 同时输出大量 2 和 9）

**最终方案**: 分位数映射（quantile mapping）。追踪 LLM interest 的经验 CDF，映射到目标 CDF。保留 LLM 的排序能力，强制边际分布。同时降 temperature 0.7→0.2，降低亲和度惩罚 -3→-1。详见 [14-50agent-validation-report.md](14-50agent-validation-report.md)。

### 3.4 Embedding 对齐 Bug（AB 方向反转）

**问题**: AB 方向验证首次运行时方向**完全反转**——个性化推荐反而比随机推荐效果更差（1/4 指标匹配）。

**根因**: `kuairand_loader.py` 中视频 embedding 使用 `hash(category) % 32` 编码品类位置，而用户 embedding 使用 `CATEGORIES.index(category)` 编码。例如 comedy 在用户侧是 index 0，在视频侧是 index 28，两者完全不对齐。dot product 算出来的不是品类匹配度，而是随机噪声。

**影响**: 个性化推荐（依赖 dot product 选视频）实际上是随机匹配，效果比真随机更差。

**修正**: 视频 embedding 改用 `CATEGORIES.index(category) % embedding_dim`。注意 `content_pool.py` 的合成视频没有此 bug。

### 3.5 Python 3.9 兼容性

**修正**: `int | None` 语法改为 `Optional[int]`。

### 3.5 DeepSeek API 支持

**实现**: LLMScheduler 添加 `provider` 参数，DeepSeek 兼容 OpenAI SDK。

---

## 四、核心技术组件

### 4.1 分位数映射校准

```python
# 目标 CDF：控制 interest 1-10 的分布
TARGET_INTEREST_CDF = [0.04, 0.08, 0.15, 0.24, 0.43, 0.58, 0.73, 0.85, 0.94, 1.0]

# 流程：LLM raw interest → 经验CDF百分位 → 目标CDF反查 → 校准后 interest
# 效果：无论 LLM 输出什么分布，最终都被标准化到目标分布
# warmup: 前30个交互不校准（数据不够建 CDF）
```

### 4.2 Beta 分布参数（当前版本）

| Band | 条件 | 参数 | 范围 | 均值 |
|------|------|------|------|------|
| Skip | interest ≤ threshold | `Exp(4), cap 0.12` | [0, 0.12] | ~0.05 |
| Band 2 | interest 6-7 | `Beta(3,2)×0.5+0.35` | [0.35, 0.85] | 0.65 |
| Band 3 | interest 8-9 | `Beta(4,1.9)×0.35+0.51` | [0.51, 0.86] | 0.74 |
| Band 4 | interest 10 | `Beta(5,2.0)×0.25+0.68` | [0.68, 0.93] | 0.86 |

### 4.3 技术栈

| 组件 | 选择 | 原因 |
|------|------|------|
| LLM | DeepSeek Chat (V3) | ¥1/M input, ¥2/M output, 兼容 OpenAI SDK |
| 异步 | asyncio + openai.AsyncOpenAI | 50 并发，~16 req/s |
| 数据 | numpy + scipy | 统计分析 |
| 配置 | PyYAML | 简单直观 |

---

## 五、成本实测

| 场景 | Agent 数 | 交互数 | 耗时 | 成本 |
|------|---------|--------|------|------|
| 冒烟测试 | 3 | 225 | 118s | ~¥0.15 |
| 快速 AB（校准后） | 100 | 768 | 34s | ~¥0.50 |
| KuaiRand 验证（v6, 20 agent） | 20 | 300 | 26s | ~¥0.35 |
| KuaiRand 验证（v12f, 50 agent） | 50 | 735 | 46s | ~¥0.35 |
| **校准全过程（54 runs）** | 20-50 | ~33,600 | ~27min | **¥18.93** |

---

## 六、KuaiRand 真实数据验证

### 数据概况

使用 KuaiRand-Pure 数据集（快手，CC BY-SA 4.0）：
- 27,285 用户 × 7,583 视频 × 118 万交互
- 12 种反馈信号：click, like, follow, comment, forward, hate, long_view, play_time_ms...
- 随机曝光数据（`is_rand=1`）：无偏 ground truth

详见 [12-kuairand-data-guide.md](12-kuairand-data-guide.md)

### 验证流程

```
KuaiRand 数据
  ├── user_features.csv → 初始化 Persona（真实活跃度/行为模式）
  ├── video_features.csv → 填充内容池（7583 真实视频）
  └── 交互日志 → 真实分布 = Ground Truth
                      ↓
              模拟分布 vs 真实分布
              Wasserstein / JS 散度 / KS 检验
```

### 关键洞察

1. **LLM 天然做二值决策**（skip 或 watch_full），无法产生平滑 watch_ratio 分布
2. **解决方案: 混合模型** — LLM 输出 interest_level (1-10)，统计模型用 Beta 分布生成 watch_ratio
3. **LLM 排序可靠、分布不可靠** — 分位数映射保留排序、强制分布，9 倍方差缩小
4. **后处理会覆盖校准** — 疲劳/亲和度惩罚在校准之后执行，需预留余量（CDF 设 43% 以达到实际 48%）
5. **先降方差再调均值** — 方差太大时参数调整在噪声中迷失方向
6. **Embedding 对齐至关重要** — 用户和视频的品类编码必须一致，否则个性化推荐等于随机匹配（导致 AB 方向反转）

---

## 七、验证状态总览

| 验证项 | 状态 | 结果 |
|--------|------|------|
| 端到端可运行 | ✅ 完成 | 3/20/50/100 agent 均跑通 |
| A/A 验证 | ✅ PASS | FPR=6%, p 值均匀分布 |
| skip_rate 对齐 | ✅ 通过 | 48.3% vs 真实 47.98%, ±2.2pp |
| completion_rate 对齐 | ✅ 通过 | 8.69% vs 真实 7.78%, ±1.2pp |
| 运行间方差 | ✅ 已解决 | skip std 从 19.5pp → 2.2pp（9倍改善） |
| avg_watch_ratio | ⚠️ 接近 | 0.414 vs 真实 0.445（差 7%） |
| like_rate | ⚠️ 偏低 | 0.11% vs 真实 0.48% |
| **AB 方向一致性** | **✅ 通过** | **4/4 指标方向正确，5/5 次运行无翻转**。详见 [docs/16](16-ab-direction-validation.md) |
| 分品类验证 | ❌ 待做 | 各品类的参与率是否分别对齐 |
| 1000 agent 缩放 | ❌ 待做 | 性能和成本验证 |
| 历史 AB 回放 | ❌ 待做 | 需要真实 AB 实验数据 |

---

## 八、下一步计划

| 优先级 | 任务 | 说明 |
|--------|------|------|
| ~~P0~~ | ~~AB 方向一致性验证~~ | ✅ 已完成，4/4 指标方向正确 |
| P1 | AB 效应量校准 | 方向正确但幅度偏差较大（如 skip 真实变化-33%，模拟-47%） |
| P1 | avg_watch_ratio 差距缩小 | 0.414 vs 0.445，可能需要提升 Band 2/3 参数 |
| P1 | like_rate 修复 | 0.11% vs 0.48%，需调 `_generate_engagement` 中的 boost 公式 |
| P1 | 分品类验证 | 确认不同品类的参与模式差异被捕捉 |
| P2 | 1000 agent 缩放 | 验证性能和成本线性扩展 |
| P3 | 用户聚类 + 人口权重 | 实现 1K→1B 缩放 |
| P3 | Fine-tune 7B 模型替代 API | 进一步降成本、提精度 |

---

## 九、文档索引

| 文档 | 内容 |
|------|------|
| docs/01-06 | 调研文档（论文、方案、框架对比） |
| docs/07 | 数据集调研 |
| docs/08-10 | OASIS/AgentTorch/Agent4Rec 深入走读 |
| docs/11 | 本文件（项目进度总览） |
| [docs/12](12-kuairand-data-guide.md) | KuaiRand 数据使用指南 |
| [docs/13](13-beta-calibration.md) | Beta 分布参数校准原理与迭代记录 |
| [docs/14](14-50agent-validation-report.md) | 50 Agent 验证报告（含分位数映射前后对比） |
| [docs/15](15-calibration-session-log.md) | 校准 session 完整操作日志（含费用） |
| [docs/16](16-ab-direction-validation.md) | AB 方向一致性验证报告（含 embedding bug 修复） |
