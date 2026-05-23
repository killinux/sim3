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
| **决策层** | Agent 怎么反应 | LLM 接收 persona+视频+记忆 → 输出 JSON 决策（watch/skip/like/comment/share） |
| **记忆层** | Agent 状态演化 | 三层记忆（工作记忆30条 + 情节记忆 + 语义摘要）+ 兴趣更新 + 疲劳累积 |
| **分析层** | 汇总结果 | 分 variant 指标计算 + Bootstrap 置信区间 + BH 多重比较修正 + A/A 验证 |

### 1.3 单次决策流程

```
1. 构建 Prompt
   System: "你是25岁男性，喜欢游戏，活跃度高..."
   User:   "下一个视频：[游戏] 30s | 5000 likes
            你最近看了: ... (最近10条)
            疲劳度: 0.35
            真实用户参与基准率: like ~5%, comment ~1%...
            请用JSON回复"

2. LLM 返回
   {"decision":"watch_full", "watch_percent":95, "like":true, ...}

3. 后处理校准
   LLM说like=true → 用persona的base_rate和watch_ratio做概率门控
   → 只有一部分like=true被保留，压到真实水平

4. 更新状态
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
| 分布对比 | 分布像不像真实？ | Wasserstein / JS 散度 | 待接入真实数据 |
| 方向一致 | 方向对不对？ | 回放历史 AB | 待历史数据 |
| 幅度准确 | 幅度准不准？ | Pearson 相关 | 待历史数据 |

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
| User Agent | `src/agents/user_agent.py` | 220 | ✅ |
| Memory | `src/agents/memory.py` | 110 | ✅ |
| Content Pool | `src/recommendation/content_pool.py` | 130 | ✅ |
| Surrogate RecSys | `src/recommendation/surrogate_recsys.py` | 100 | ✅ |
| Simulation Engine | `src/simulation/engine.py` | 190 | ✅ |
| LLM Scheduler | `src/simulation/scheduler.py` | 130 | ✅ |
| Metrics | `src/analysis/metrics.py` | 70 | ✅ |
| Treatment Effect | `src/analysis/treatment_effect.py` | 120 | ✅ |
| Distribution Metrics | `src/validation/distribution_metrics.py` | 90 | ✅ |
| A/A Test | `src/validation/aa_test.py` | 70 | ✅ |
| Run Script | `scripts/run_simulation.py` | 150 | ✅ |

### 2.3 运行验证结果

#### 冒烟测试（3 agent）

| 指标 | 值 |
|------|---|
| 交互数 | 225 |
| 耗时 | 118s |
| 错误数 | 0 |
| 成本 | ~$0.02 |

#### 校准前（100 agent, AB test）

| 指标 | Control | Treatment | 问题 |
|------|---------|-----------|------|
| like_rate | 48.6% | 54.8% | ⚠️ 远高于真实 ~5% |
| comment_rate | 0% | 2% | 偏低 |
| skip_rate | 43.8% | 38.2% | 合理 |
| completion_rate | 50.0% | 55.3% | 合理 |
| A/A 验证 | PASS | | ✅ |

#### 校准后（100 agent, AB test）—— 当前版本

| 指标 | Control | Treatment | 变化 | p 值 |
|------|---------|-----------|------|------|
| **like_rate** | **15.4%** | **22.0%** | +63.8% | 0.125 |
| comment_rate | 0% | 0% | - | - |
| skip_rate | 63.7% | 49.4% | -19.9% | **0.023** |
| completion_rate | 32.4% | 45.8% | +51.5% | **0.043** |
| avg_watch_ratio | 0.341 | 0.478 | +50.7% | **0.033** |
| A/A 验证 | PASS | | ✅ |
| 耗时 | **34 秒** | | |
| 成本 | **~$0.06** | | |

**AB 结论**: 高多样性推荐（treatment）显著降低跳过率（p=0.023）、提升完播率（p=0.043）和观看比例（p=0.033），方向合理。

---

## 三、已修正的问题

### 3.1 Like Rate 过高（52% → 15-22%）

**问题**: 首次运行 like_rate 达 52%，真实平台约 5%。

**根因**: LLM 的 RLHF 训练使其倾向"友好"和"积极"，在论文中被称为 **"正向偏差"**（OmniBehavior, 2026）。具体表现：
- 评分中心化：偏向中高分
- 曝光偏差：过度服从推荐
- 动作简化：缺乏真实人类的犹豫和拒绝

**修正方案（两层）**:

**第一层 - Prompt 校准**: 在 prompt 中注入真实参与率基准

```
IMPORTANT - Real user engagement benchmarks (be realistic, not generous):
- Most users SKIP 40-60% of videos after watching <2 seconds
- Only 5-8% of watched videos get a LIKE
- Only 0.5-2% get a COMMENT
- Only 0.3-1% get a SHARE
- Users are selective and picky, not enthusiastic about everything
```

**第二层 - 后处理校准**: 即使 LLM 仍返回 like=true，通过概率门控压到合理范围

```python
def _calibrate_engagement(self, liked, commented, shared, followed, watch_ratio, video):
    # 1. 如果观看比例 < 30%，清除所有参与动作
    if watch_ratio < 0.3:
        return False, False, False, False

    # 2. 用 persona 的 base_rate × 观看参与度 做概率门控
    #    LLM 说 like=true 时，只有 base_rate 概率真正保留
    if liked:
        keep_prob = base_like_rate × engagement_mult / cap
        liked = random() < keep_prob
```

**效果**: like_rate 从 52% 降到 15-22%，仍偏高但已在可工作范围。进一步降到 5% 需要微调（SFT+DPO）。

### 3.2 Python 3.9 兼容性

**问题**: 代码使用了 `int | None` 语法（Python 3.10+），在 Python 3.9 上 TypeError。

**修正**: 改为无类型注解的默认参数 `override_agents=None`。

### 3.3 OpenAI SDK 版本

**问题**: 环境中 openai 0.27.x 没有 `AsyncOpenAI`。

**修正**: 升级到 openai 2.38.0。

### 3.4 DeepSeek API 支持

**需求**: 用 DeepSeek 降低成本（比 GPT-4o-mini 便宜 53%）。

**实现**: LLMScheduler 添加 `provider` 参数和 `PROVIDER_CONFIGS` 字典，DeepSeek 兼容 OpenAI SDK，只需改 `base_url`。

---

## 四、技术栈

| 组件 | 选择 | 原因 |
|------|------|------|
| LLM | DeepSeek Chat (V3) | $0.27/M input, $1.10/M output, 兼容 OpenAI SDK |
| 异步 | asyncio + openai.AsyncOpenAI | 50 并发，22 req/s |
| 数据 | numpy + scipy | 统计分析 |
| 配置 | PyYAML | 简单直观 |

---

## 五、成本实测

| 场景 | Agent 数 | 交互数 | 耗时 | Token | 成本 |
|------|---------|--------|------|-------|------|
| 冒烟测试 | 3 | 225 | 118s | 129K | ~$0.02 |
| 快速 AB（校准前） | 100 | 768 | 59s | 375K | ~$0.06 |
| 快速 AB（校准后） | 100 | 768 | 34s | 449K | ~$0.07 |
| KuaiRand 验证 | 20 | 300 | 26s | ~350K | ~$0.05 |

**DeepSeek prompt caching 生效**: 第二次运行速度从 59s 降到 34s（cache hit 降低延迟）。

---

## 五-B、KuaiRand 真实数据验证

### 数据概况

使用 KuaiRand-Pure 数据集（快手，CC BY-SA 4.0）：
- 27,285 用户 × 7,583 视频 × 118 万交互
- 12 种反馈信号：click, like, follow, comment, forward, hate, long_view, play_time_ms...
- 随机曝光数据（`is_rand=1`）：无偏 ground truth

### 验证流程

```
KuaiRand 数据
  ├── user_features.csv → 初始化 20 个 Persona（真实活跃度/行为模式）
  ├── video_features.csv → 填充内容池（7583 真实视频）
  └── 交互日志 → 真实分布 = Ground Truth
                      ↓
              模拟分布 vs 真实分布
              Wasserstein / JS 散度
```

### 验证结果（v1 基线）

| 指标 | 快手真实 | 模拟 | 差距 | 说明 |
|------|---------|------|------|------|
| like_rate | 0.48% | 0.00% | 校准过强 | 需调回 |
| comment_rate | 0.03% | 0.00% | 一致 | OK |
| skip_rate | 47.98% | 79.67% | +66% | LLM 过度保守 |
| completion_rate | 7.78% | 15.67% | +101% | 偏高 |
| avg_watch_ratio | 0.445 | 0.172 | -61% | 偏低 |
| **Wasserstein 距离** | - | **0.218** | - | 基线值 |
| **JS 散度** | - | **0.097** | - | 基线值 |

### 分析

1. Prompt 加了"be realistic, not generous"后 LLM 变得过度保守
2. 校准层把 like 从 52% 压到了 0%（过度修正）
3. 跳过率 80% 远高于真实的 48%
4. 需要在"LLM 太友好"和"LLM 太保守"之间找到平衡

### 后续优化方向

- 调节 prompt 中的基准率描述（5-8% → 更精确的分品类基准）
- 校准层参数调优（降低门控强度）
- 用 KuaiRand 的真实参与率做 per-persona 的 base_rate 初始化

---

## 六、下一步计划

| 优先级 | 任务 | 预期效果 |
|--------|------|---------|
| P0 | 校准层参数调优（like 从 0% 调回 1-5%） | Wasserstein 下降 |
| P0 | 调节 prompt 平衡点（不太友好也不太保守） | skip_rate 接近 48% |
| P1 | 扩大到 100 agent + KuaiRand 数据 | 更可靠的统计 |
| P1 | 分品类验证（gaming/pets/food 各自的参与率） | 细粒度校准 |
| P2 | 用户聚类 + 人口权重 | 实现 1K→1B 缩放 |
| P2 | 推荐系统在线更新（反馈闭环） | 捕捉"推荐塑造偏好"动态 |
| P3 | Fine-tune 7B 模型替代 API | 进一步降成本、提精度 |
