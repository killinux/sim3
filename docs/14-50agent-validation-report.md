# 50 Agent 验证报告

> 日期：2026-05-24
> 数据集：KuaiRand-Pure（快手，1,186,059 条真实交互）

## 一、实验配置

| 参数 | 值 |
|------|-----|
| Agent 数量 | 50 |
| 每 Agent 会话数 | 1 |
| 每会话最大视频数 | 15 |
| LLM | DeepSeek Chat (deepseek-chat) |
| Temperature | 0.2 |
| 并发数 | 32 |
| 随机种子 | 42 |
| Persona 来源 | KuaiRand 真实用户特征 |
| 内容池 | KuaiRand 真实视频 7,583 条 |
| 单次耗时 | ~46s |
| 单次交互数 | ~735 |

## 二、校准前基线（v11，无 calibration layer）

5 次运行，temperature=0.4，无分位数映射：

| 指标 | 均值 | 标准差 | 真实值 | 判断 |
|------|------|--------|--------|------|
| skip_rate | 53.5% | **19.5pp** | 47.98% | ⚠️ 偏高，方差极大 |
| completion_rate | 7.32% | 4.1pp | 7.78% | ✅ 均值好，方差大 |
| avg_watch_ratio | 0.376 | **0.124** | 0.445 | ⚠️ 偏低，方差大 |
| like_rate | 0.30% | 0.26pp | 0.48% | ⚠️ 偏低 |

**核心问题**：LLM interest_level 输出的 run-to-run 方差主导了所有指标的不稳定。5 次 skip_rate 从 19% 到 73%。

## 三、分位数映射校准（v12）

### 3.1 方法

引入 **quantile mapping calibration layer**：追踪 LLM interest_level 的经验 CDF，通过目标 CDF 做分位数映射，将 LLM 输出的任意分布标准化到预设目标分布。

```python
# 目标 CDF：interest 1-10 的累积分布
TARGET_INTEREST_CDF = [0.04, 0.08, 0.15, 0.24, 0.43, 0.58, 0.73, 0.85, 0.94, 1.0]

# 校准流程
raw_interest = LLM_output           # 1-10
percentile = empirical_CDF(raw)     # LLM实际分布中的分位数
interest = inverse_target_CDF(percentile)  # 映射到目标分布
```

**原理**：LLM 的排序能力（哪个视频更有趣）是可靠的，但绝对值分布不稳定。分位数映射保留排序、强制分布，从根本上消除 LLM 的 batch-level bias。

### 3.2 目标分布设计

| Interest | 目标占比 | Band | 说明 |
|----------|---------|------|------|
| 1-5 | 43% | Skip | 快速划走 |
| 6 | 15% | Band 2 | 看了一部分 |
| 7 | 15% | Band 2 | 看了较多 |
| 8 | 12% | Band 3 | 比较感兴趣 |
| 9 | 9% | Band 3 | 高度感兴趣 |
| 10 | 6% | Band 4 | 非常感兴趣 |

注意：目标 skip 占比为 43%（低于真实 48%），因为疲劳/低亲和度惩罚会在后处理中额外推入约 5% 的 skip。

### 3.3 其他同步调整

| 调整项 | 旧值 | 新值 | 原因 |
|--------|------|------|------|
| temperature | 0.4 | 0.2 | 降低 LLM 基础方差 |
| Band 2 watch_ratio | `Beta(3,2)*0.6+0.2` | `Beta(3,2)*0.5+0.35` | 提升 avg_wr，mean 0.56→0.65 |
| 低亲和度惩罚 | interest - 3 | interest - 1 | 防止校准后的 interest 被大幅覆盖 |

## 四、校准后结果

### 4.1 5 次运行原始数据

| 指标 | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | **均值** | **标准差** | 真实值 |
|------|-------|-------|-------|-------|-------|---------|----------|--------|
| skip_rate | 48.1% | 47.8% | 44.5% | 50.5% | 50.4% | **48.3%** | **2.2pp** | 47.98% |
| completion_rate | 8.81% | 10.6% | 9.07% | 6.93% | 8.02% | **8.69%** | **1.2pp** | 7.78% |
| avg_watch_ratio | 0.417 | 0.423 | 0.432 | 0.396 | 0.402 | **0.414** | **0.015** | 0.445 |
| like_rate | 0.27% | 0.27% | 0.00% | 0.00% | 0.00% | **0.11%** | 0.15pp | 0.48% |

### 4.2 校准前后对比

| 指标 | 校准前均值 | 校准前标准差 | **校准后均值** | **校准后标准差** | 真实值 | 方差缩小倍数 |
|------|-----------|------------|-------------|--------------|--------|------------|
| skip_rate | 53.5% | 19.5pp | **48.3%** | **2.2pp** | 47.98% | **9x** |
| completion_rate | 7.32% | 4.1pp | **8.69%** | **1.2pp** | 7.78% | **3x** |
| avg_watch_ratio | 0.376 | 0.124 | **0.414** | **0.015** | 0.445 | **8x** |

### 4.3 均值对齐评估

| 指标 | 校准后均值 | 真实值 | 偏差 | 判断 |
|------|-----------|--------|------|------|
| skip_rate | 48.3% | 47.98% | +0.3pp | ✅ 几乎完美 |
| completion_rate | 8.69% | 7.78% | +0.9pp | ✅ 良好 |
| avg_watch_ratio | 0.414 | 0.445 | -7.0% | ⚠️ 略低 |
| like_rate | 0.11% | 0.48% | -77% | ⚠️ 偏低 |

## 五、迭代过程摘要

分位数映射的 CDF 和辅助参数经过多轮迭代：

| 版本 | 改动 | skip 均值/std | completion 均值/std | avg_wr | 判断 |
|------|------|-------------|-------------------|--------|------|
| v11 | 基线（无calibration） | 53.5%/19.5pp | 7.32%/4.1pp | 0.376 | 方差太大 |
| v12a | +affinity prior blending | 47.6%/~20pp | 1.08%/- | 0.348 | completion崩了 |
| v12b | +mean correction | 47.0%/14.9pp | 11.0%/- | 0.427 | 双峰分布无效 |
| v12c | +quantile mapping (CDF v1) | **48.1%/2.4pp** | 4.98%/2.6pp | 0.375 | ✅ 方差解决, completion低 |
| v12d | CDF v2 (更多Band 3/4) | 52.6%/3.6pp | 7.93%/1.6pp | 0.388 | completion好, skip偏高 |
| v12e | CDF v3 + 减skip比例 | 51.5%/6.4pp | 8.15%/2.4pp | 0.393 | 改善有限 |
| **v12f** | **+减低亲和度惩罚** | **48.3%/2.2pp** | **8.69%/1.2pp** | **0.414** | **✅ 最优** |

## 六、仍存在的问题

### 6.1 avg_watch_ratio 偏低 7%

均值 0.414 vs 真实 0.445。推算真实数据中非 skip 交互的平均 watch_ratio 为 0.81，而模拟中 Band 2 均值仅 0.65。可能需要进一步提升 Band 2/3 的 Beta 参数。

### 6.2 like_rate 偏低

均值 0.11% vs 真实 0.48%。`_generate_engagement` 中 `boost = (interest/5)^1.5 × watch_ratio` 的乘性结构可能在中低 interest 时过度压低。需要单独调优 boost 公式或 base_rate。

### 6.3 warmup 期

分位数映射在前 30 个交互内不生效（不够数据构建经验 CDF）。50 agent 并行时，前 30 个交互大约在前 1-2 轮 LLM 调用内完成，影响有限。

## 七、总结

| 维度 | 状态 | 说明 |
|------|------|------|
| skip_rate 对齐 | ✅ 通过 | 48.3% vs 真实 47.98%，±2.2pp |
| completion_rate 对齐 | ✅ 通过 | 8.69% vs 真实 7.78%，±1.2pp |
| avg_watch_ratio 对齐 | ⚠️ 可接受 | 0.414 vs 真实 0.445，差 7% |
| 参与率量级 | ⚠️ like偏低 | like 0.11% vs 0.48%，需调 boost 公式 |
| **运行间方差** | **✅ 已解决** | skip std 从 19.5pp 降至 2.2pp（9倍改善） |

**分位数映射是本轮最关键的改进**，从根本上解决了 LLM 输出不确定性导致的方差问题。系统现在可以稳定复现，为 AB 方向一致性验证奠定了基础。

## 八、代码位置

| 内容 | 文件 | 说明 |
|------|------|------|
| 分位数映射 | `src/agents/user_agent.py` | `calibrate_interest()` 类方法 |
| 目标 CDF | `src/agents/user_agent.py` | `TARGET_INTEREST_CDF` 类变量 |
| Tracker 重置 | `src/simulation/engine.py` | `run()` 开头调用 `reset_interest_tracker()` |
| Beta 参数 | `src/agents/user_agent.py` | `parse_llm_response()` 中分区采样 |
| 验证脚本 | `scripts/validate_with_kuairand.py` | 50 agent, temperature 0.2 |
