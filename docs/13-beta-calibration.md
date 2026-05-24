# Beta 分布参数校准记录

## 一、问题背景

### 1.1 为什么需要 Beta 分布

LLM 天然做二值决策（"看" 或 "跳过"），无法直接产生平滑的 `watch_ratio` 分布。如果直接让 LLM 输出 `watch_percent`，结果会高度集中在 0%、50%、100% 三个整数点上，与真实用户的连续分布严重不符。

因此我们采用**混合架构**：

```
LLM → interest_level (1-10)
           ↓
    统计模型分区采样
           ↓
  Beta(α,β) × scale + offset → watch_ratio ∈ [0, 1]
```

LLM 负责"理解力"（这个视频对这个用户多有趣），统计模型负责"生成力"（把理解映射到符合真实分布的观看比例）。

### 1.2 Beta 分布简介

Beta(α, β) 分布定义在 [0, 1] 上，由两个参数控制形状：

- **α (alpha)**: 越大，分布越向右偏（值越大）
- **β (beta)**: 越大，分布越向左偏（值越小）
- **均值** = α / (α + β)
- **方差**随 α+β 增大而减小（分布越集中）

通过 `Beta(α,β) × scale + offset` 变换，可以把分布从 [0,1] 映射到任意 [offset, offset+scale] 区间。

```
例：Beta(4, 2) × 0.35 + 0.51
    均值 = 4/6 × 0.35 + 0.51 = 0.74
    范围 = [0.51, 0.86]
```

### 1.3 校准目标

对齐 KuaiRand（快手）真实数据的关键指标：

| 指标 | 真实值 | 含义 |
|------|--------|------|
| completion_rate | 7.78% | watch_ratio ≥ 0.8 的比例 |
| skip_rate | 47.98% | watch_ratio < 0.15 的比例（2秒内划走） |
| avg_watch_ratio | 0.4454 | 平均观看比例 |
| like_rate | 0.48% | 点赞率 |

## 二、分区机制

### 2.1 Interest → Watch Ratio 映射

LLM 输出 `interest_level` (1-10)，经过疲劳惩罚和品类亲和度调整后得到 `effective_interest`，然后分区采样：

```python
# 1. 疲劳惩罚
fatigue_penalty = fatigue * 0.3
effective_interest = max(1, interest - fatigue_penalty * 10)

# 2. 低亲和度品类额外惩罚
if category_affinity < 0.02:
    effective_interest = max(1, effective_interest - 3)

# 3. 动态 skip 阈值（品类亲和度越高，越不容易 skip）
skip_threshold = 5 - int(category_affinity * 5)  # clamp [3, 6]

# 4. 分区采样
if effective_interest <= skip_threshold:     → Skip Band
elif effective_interest <= 7:                → Band 2 (Partial Watch)
elif effective_interest <= 9:                → Band 3 (High Interest)
else (== 10):                                → Band 4 (Very High Interest)
```

### 2.2 各 Band 的 Beta 参数

| Band | 条件 | 参数 | 范围 | 均值 | 含义 |
|------|------|------|------|------|------|
| Skip | interest ≤ threshold | `Exp(4), cap 0.12` | [0, 0.12] | ~0.05 | 快速划走，几乎不看 |
| Band 2 | interest 6-7 | `Beta(3,2)×0.6+0.2` | [0.2, 0.8] | 0.56 | 看了一部分，没看完 |
| Band 3 | interest 8-9 | `Beta(4,1.9)×0.35+0.51` | [0.51, 0.86] | 0.74 | 比较感兴趣，偶尔看完 |
| Band 4 | interest 10 | `Beta(5,2.0)×0.25+0.68` | [0.68, 0.93] | 0.86 | 非常感兴趣，大概率看完 |

## 三、校准迭代过程

### 3.1 v6 基线（校准前）

参数：
- Band 3: `Beta(4, 1.5) × 0.35 + 0.55` → 范围 [0.55, 0.90]
- Band 4: `Beta(6, 1.5) × 0.2 + 0.8` → 范围 [0.80, 1.00]
- skip_threshold: `6 - int(affinity×5)`, clamp [4, 7]
- temperature: 0.7

结果（50 agent）：

| 指标 | 模拟值 | 真实值 | 判断 |
|------|--------|--------|------|
| completion_rate | **20.8%** | 7.78% | 偏高 2.67x ❌ |
| skip_rate | 56% | 47.98% | 偏高 ⚠️ |
| avg_watch_ratio | 0.393 | 0.4454 | 偏低 12% ⚠️ |

**诊断**：completion 偏高的根因是 Band 4 的最小值就是 0.8，导致 interest=10 的视频 **100% 完播**；Band 3 有约 50% 概率完播（β=1.5 太小，分布过度右偏）。

### 3.2 v7 — 激进下调 Beta

改动：
- Band 3: β 从 1.5 → **2.0**，offset 从 0.55 → **0.5**
- Band 4: α 从 6 → **4**，β 从 1.5 → **2.0**，scale/offset 重构为 `×0.3+0.65`

结果（20 agent, 单次）：
- completion_rate: **4.33%** → 矫枉过正，低于目标

**教训**：Band 4 从 [0.8,1.0] 一下子降到 [0.65,0.95]，变化过大。

### 3.3 v8 — 回调参数

改动：
- Band 3: β 回到 **1.7**，offset 提到 **0.53**
- Band 4: α 提到 **5**，β=**1.8**，改为 `×0.25+0.72`

结果（20 agent, 3次均值, temperature 0.7）：
- completion_rate: **14.14%** → 还是偏高
- 3次方差极大：skip_rate 从 34% 到 77%

**教训**：temperature 0.7 + 20 agent 方差太大，无法有效调参。

### 3.4 v9 — 降低 LLM 温度

改动：
- temperature: 0.7 → **0.4**
- Band 3: 回到 `Beta(4, 2.0)×0.35+0.5`
- Band 4: `Beta(5, 2.0)×0.25+0.68`

结果（20 agent, 3次均值, temperature 0.4）：
- completion_rate: **3.56%** → 太低
- skip_rate: **59.3%** → 太高
- avg_watch_ratio: **0.330** → 太低

**诊断**：低温度下 LLM 输出 interest 集中在 5-6，大量命中 skip（旧阈值=6），skip 占比过高拉低了所有指标。

### 3.5 v10 — 降低 Skip 阈值

改动：
- skip_threshold: `6 - int(affinity×5)` → `5 - int(affinity×5)`，clamp [4,7] → [3,6]
- Band 3: 同时调 β 到 **1.6** 试探上界

结果（3次均值）：
- completion_rate: **16.89%** → skip 好了但 completion 反弹
- skip_rate: **48.2%** → 接近目标 ✅

**诊断**：skip 阈值降 1 后效果显著，但 Band 3 的 β=1.6 过小导致完播概率反弹到 ~25%。

### 3.6 v11 — 最终参数（当前版本）

改动：
- Band 3: β 从 1.6 → **1.9**，offset 从 0.53 → **0.51**

最终参数：
```python
# Skip Band: interest ≤ skip_threshold
watch_ratio = max(0.0, rng.expovariate(4.0))
watch_ratio = min(watch_ratio, 0.12)

# Band 2: interest 6-7
watch_ratio = rng.betavariate(3.0, 2.0) * 0.6 + 0.2

# Band 3: interest 8-9
watch_ratio = rng.betavariate(4.0, 1.9) * 0.35 + 0.51

# Band 4: interest 10
watch_ratio = rng.betavariate(5.0, 2.0) * 0.25 + 0.68

# Skip threshold
skip_threshold = 5 - int(category_affinity * 5)  # clamp [3, 6]
```

结果（20 agent, 3次均值, temperature 0.4）：

| 指标 | 3次均值 | 真实值 | 差距 |
|------|---------|--------|------|
| completion_rate | **8.23%** | 7.78% | +0.45pp ✅ |
| avg_watch_ratio | **0.432** | 0.445 | -3% ✅ |
| skip_rate | **43.8%** | 47.98% | -4pp ✅ |
| like_rate | **0.11%** | 0.48% | 样本量问题 ⚠️ |

单次最优（Run 3）：completion 8.70%, avg_wr **0.4461** vs 真实 0.4454。

## 四、分位数映射校准（v12，方差解决方案）

### 4.1 问题

v11 的 20 agent 验证均值接近目标，但扩到 50 agent 后发现 **run-to-run 方差极大**：skip_rate 标准差 19.5pp（5 次从 19% 到 73%）。根因是 LLM 的 interest_level 输出存在 batch-level correlation——同一次运行中，LLM 要么集中给高 interest，要么集中给低 interest。

### 4.2 尝试过的失败方案

| 方案 | 原理 | 结果 | 失败原因 |
|------|------|------|---------|
| Affinity prior blending | 混入确定性的品类亲和度先验 | completion 降至 ~1% | 先验太低（大部分亲和度 <0.1），砍掉了高 interest 尾部 |
| Mean correction | 追踪 interest 均值，动态修正偏移 | 方差改善有限 | 无法修复双峰分布（LLM 同时输出很多 2 和 9，均值正确但分布错误） |

### 4.3 最终方案：分位数映射

**核心思想**：LLM 的**排序**能力可靠（视频 A 比 B 更有趣），但**绝对值**分布不稳定。分位数映射保留排序、强制边际分布。

```python
TARGET_INTEREST_CDF = [0.04, 0.08, 0.15, 0.24, 0.43, 0.58, 0.73, 0.85, 0.94, 1.0]

def calibrate_interest(raw_interest):
    # 1. 更新经验 CDF
    interest_counts[raw - 1] += 1
    # 2. 计算 raw 在经验分布中的百分位
    percentile = empirical_cdf_midpoint(raw)
    # 3. 通过目标 CDF 反查：percentile → 校准后 interest
    return inverse_target_cdf(percentile)
```

### 4.4 同步调整

| 调整 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| Temperature | 0.4 | 0.2 | 降低 LLM 基础方差 |
| Band 2 | `Beta(3,2)*0.6+0.2` | `*0.5+0.35` | 提升 avg_wr |
| 低亲和度惩罚 | -3 | -1 | 防止校准被后处理覆盖 |

### 4.5 效果

| 指标 | 校准前 (std) | 校准后 (std) | 真实值 | 方差缩小 |
|------|-------------|-------------|--------|---------|
| skip_rate | 53.5% (19.5pp) | **48.3%** (2.2pp) | 47.98% | **9x** |
| completion | 7.32% (4.1pp) | **8.69%** (1.2pp) | 7.78% | **3x** |
| avg_wr | 0.376 (0.124) | **0.414** (0.015) | 0.445 | **8x** |

## 五、关键发现与原则

### 5.1 参数灵敏度排序

1. **分位数映射 CDF**：影响最大的"元参数"。直接控制 interest 分布，决定各 Band 的占比
2. **skip_threshold**：在 CDF 固定的情况下，仍是 skip_rate 的主要控制旋钮
3. **Band 3/4 的 Beta β 参数**：对 completion_rate 最敏感
4. **低亲和度惩罚**：影响 CDF 校准的有效性（惩罚太大会覆盖校准结果）

### 5.2 调参方法论

1. **先降方差再调均值**：temperature 和分位数映射是方差的主要控制手段
2. **分位数映射 > 均值修正**：均值修正无法处理双峰分布，分位数映射从根本上解决
3. **逐参数二分法**：每次只改一个参数，5 次均值判断方向
4. **注意后处理覆盖**：校准发生在 raw interest 级别，疲劳/亲和度惩罚在后面可能覆盖校准效果

### 5.3 仍存在的问题

- **avg_watch_ratio 差 7%**：0.414 vs 0.445，可能需要提升 Band 2/3 的 Beta 参数
- **like_rate 偏低**：0.11% vs 0.48%，需调 `_generate_engagement` 中的 boost 公式
- **warmup 期**：前 30 个交互未校准，50 agent 并行时影响约 4%

## 六、代码位置

| 内容 | 文件 |
|------|------|
| 分位数映射 | `src/agents/user_agent.py` — `calibrate_interest()` |
| 目标 CDF | `src/agents/user_agent.py` — `TARGET_INTEREST_CDF` |
| Beta 分布采样 | `src/agents/user_agent.py` — `parse_llm_response()` |
| Tracker 重置 | `src/simulation/engine.py` — `run()` |
| 验证脚本 | `scripts/validate_with_kuairand.py` |
