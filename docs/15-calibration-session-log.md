# 校准迭代完整记录（2026-05-24）

## 一、本次目标

1. 调 Beta 分布参数压低 completion_rate（20.8% → 7.78%）
2. 先用 20 agent 验证，再扩到 50 agent
3. 解决发现的 LLM 方差问题

## 二、操作时间线

### Phase 1：Beta 参数校准（20 agent）

| 时间 | 版本 | 改动 | 结果 | 判断 |
|------|------|------|------|------|
| 14:25 | v7 | Band 3: β 1.5→2.0, offset 0.55→0.5<br>Band 4: α 6→4, β 1.5→2.0, 重构为 ×0.3+0.65 | completion 4.33% | 矫枉过正 |
| 14:27 | v8 | Band 3: β→1.7, offset→0.53<br>Band 4: α→5, β→1.8, ×0.25+0.72 | skip 77%, completion 6.08%<br>（单次，方差大） | 不稳定 |
| 14:29 | v8×3 | 同 v8，跑3次 | completion 均值 14.14%<br>方差极大（skip 34-77%） | 方差是主要问题 |
| 14:31 | v9 | Band 3: 回到 β=2.0, ×0.35+0.5<br>Band 4: ×0.25+0.68<br>**temperature 0.7→0.4** | completion 3.56%<br>skip 59.3% | skip 偏高 |
| 14:33 | v10 | **skip_threshold 6→5**, clamp [4,7]→[3,6]<br>Band 3: β→1.6, offset→0.53 | completion 16.89%<br>**skip 48.2%** ✅ | skip 好了,<br>completion 反弹 |
| 14:35 | v11 | Band 3: β 1.6→**1.9**, offset→0.51 | completion **8.23%**<br>skip 43.8%<br>avg_wr 0.432 | ✅ 20 agent<br>校准成功 |

**Phase 1 耗时**：~15 分钟，6 轮迭代
**Phase 1 总跑数**：14 次（1+1+3+3+3+3）
**Phase 1 总 LLM 调用**：14 × ~300 = **~4,200 次**

### Phase 2：50 Agent 扩展验证

| 时间 | 版本 | 改动 | 结果 | 判断 |
|------|------|------|------|------|
| 17:27 | v11×5 | 扩到 50 agent，跑5次 | completion 均值 7.32% ✅<br>skip **std 19.5pp** ❌ | 均值好，方差灾难 |

**发现关键问题**：50 agent 方差没有比 20 agent 收敛，skip_rate 从 19% 到 73%。证明方差来自 LLM batch-level correlation，不是统计采样。

### Phase 3：方差解决（分位数映射）

| 时间 | 版本 | 改动 | 结果 | 判断 |
|------|------|------|------|------|
| 17:40 | v12a | +affinity prior blending<br>(0.55×LLM + 0.45×prior)<br>temp→0.2 | completion ~1%<br>方差未改善 | ❌ prior 太低<br>砍掉高 interest 尾部 |
| 17:46 | v12b | 改为 mean correction<br>(追踪均值，动态修正)<br>TARGET_MEAN=5.8 | completion 11%<br>skip std 14.9pp | ⚠️ 方差有改善<br>但均值偏移 |
| 17:52 | v12b' | TARGET_MEAN 5.8→5.5<br>correction 0.6→0.8 | skip std 16.7pp | ❌ 更差了<br>双峰分布无法修正 |
| 17:58 | **v12c** | **改为分位数映射**<br>TARGET_CDF=[...0.48, 0.65, 0.80...] | skip **48.1% ±2.4pp** ✅<br>completion 4.98% | ✅ 方差解决！<br>completion 偏低 |
| 18:03 | v12d | CDF 调整：Band 3/4 占比增加<br>Band 2: ×0.6+0.2 → ×0.5+0.35 | skip 52.6% ±3.6pp<br>completion 7.93% | completion 好了<br>skip 偏高 |
| 18:09 | v12e | CDF: skip 目标 48%→43%<br>(预留 fatigue/affinity 余量) | skip 51.5% ±6.4pp | 改善有限 |
| 18:14 | **v12f** | **低亲和度惩罚 -3→-1** | skip **48.3% ±2.2pp**<br>completion **8.69% ±1.2pp**<br>avg_wr **0.414 ±0.015** | **✅ 最终版本** |

**Phase 3 耗时**：~40 分钟，7 轮迭代
**Phase 3 总跑数**：35 次（7 × 5 次均值）
**Phase 3 总 LLM 调用**：35 × ~735 = **~25,725 次**

## 三、最终结果

### 3.1 校准前后对比

| 指标 | 基线 (v6) | 校准后 (v12f) | 真实值 | 改善 |
|------|-----------|-------------|--------|------|
| completion_rate | 20.8% | **8.69% ±1.2pp** | 7.78% | ✅ 从 2.67x 偏差 → 0.9pp 偏差 |
| skip_rate | 56% | **48.3% ±2.2pp** | 47.98% | ✅ 从 8pp 偏差 → 0.3pp 偏差 |
| avg_watch_ratio | 0.393 | **0.414 ±0.015** | 0.445 | ⚠️ 从 12% 差距 → 7% 差距 |
| like_rate | 0.27% | **0.11%** | 0.48% | ⚠️ 待改善 |
| skip_rate std | ~20pp | **2.2pp** | - | ✅ 方差缩小 9 倍 |
| completion std | ~4pp | **1.2pp** | - | ✅ 方差缩小 3 倍 |

### 3.2 最终参数

```python
# 分位数映射目标 CDF
TARGET_INTEREST_CDF = [0.04, 0.08, 0.15, 0.24, 0.43, 0.58, 0.73, 0.85, 0.94, 1.0]
_CALIBRATION_WARMUP = 30

# Beta 参数
Skip:    Exp(4), cap 0.12
Band 2:  Beta(3, 2.0) × 0.5 + 0.35    # mean 0.65
Band 3:  Beta(4, 1.9) × 0.35 + 0.51   # mean 0.74
Band 4:  Beta(5, 2.0) × 0.25 + 0.68   # mean 0.86

# Skip 阈值
skip_threshold = 5 - int(category_affinity * 5)  # clamp [3, 6]

# 低亲和度惩罚
if category_affinity < 0.02: effective_interest -= 1  # (原来是 -3)

# LLM
temperature = 0.2
```

## 四、费用统计

### 4.1 LLM API 调用量

| 阶段 | Agent 数 | 跑数 | 每跑交互数 | 总 LLM 调用 |
|------|---------|------|-----------|------------|
| Phase 1 (20 agent) | 20 | 14 | ~300 | ~4,200 |
| Phase 2 (50 agent baseline) | 50 | 5 | ~735 | ~3,675 |
| Phase 3 (方差解决) | 50 | 35 | ~735 | ~25,725 |
| **合计** | - | **54** | - | **~33,600** |

### 4.2 Token 用量（基于最后一次运行的单位消耗推算）

单次 50 agent 运行实测：
- 输入 token：325,141（~442/次交互）
- 输出 token：43,954（~60/次交互）
- 合计：369,095 token/run

| 阶段 | 输入 token | 输出 token | 合计 |
|------|-----------|-----------|------|
| Phase 1 (14 runs × 20 agent) | ~1.85M | ~252K | ~2.1M |
| Phase 2 (5 runs × 50 agent) | ~1.63M | ~220K | ~1.85M |
| Phase 3 (35 runs × 50 agent) | ~11.38M | ~1.54M | ~12.92M |
| **合计** | **~14.86M** | **~2.01M** | **~16.87M** |

### 4.3 DeepSeek API 费用

DeepSeek Chat 定价（2026）：
- 输入：¥1/M tokens（cache miss），¥0.1/M（cache hit）
- 输出：¥2/M tokens

实际费用（DeepSeek 账单）：

| 项目 | 费用 |
|------|------|
| **DeepSeek API 总计** | **¥18.93 (~$2.63)** |
| 估算 token 总量 | ~16.87M |
| 估算单次 50-agent 运行 | ~¥0.35 |

### 4.4 其他资源

| 项目 | 消耗 |
|------|------|
| 总运行时间 | 54 runs × ~30s avg = ~27 分钟计算时间 |
| 墙钟时间 | ~2 小时（含分析和编码） |
| 网络流量 | ~50MB（API 请求/响应） |
| 磁盘 | 输出报告 < 1MB |

## 五、关键决策点记录

### 决策 1：先降方差还是先调均值？
- **选择**：先降方差
- **原因**：temperature 0.7 下 3 次跑结果完全不同，无法判断参数调整的方向。必须先让结果可复现
- **验证**：temperature 0.4 后方差减半，才能有效调参

### 决策 2：均值修正 vs 分位数映射？
- **选择**：分位数映射
- **原因**：均值修正无法处理双峰分布（LLM 同时输出很多 2 和 9，均值正确但 skip_rate 极端）。分位数映射强制整个分布形状
- **验证**：skip_rate std 从 19.5pp 直降到 2.4pp

### 决策 3：CDF 中 skip 占比设多少？
- **选择**：43%（而非真实值 48%）
- **原因**：疲劳惩罚和低亲和度惩罚会在 CDF 校准后额外推入 ~5% 的 skip
- **验证**：CDF 43% → 实际 skip_rate 48.3%，精确命中

### 决策 4：低亲和度惩罚 -3 → -1
- **选择**：减弱惩罚
- **原因**：-3 惩罚太大，把 CDF 校准后的 interest 7-8 直接打回 skip，导致校准失效
- **验证**：改为 -1 后 skip_rate 从 51.5% 降到 48.3%，方差从 6.4pp 降到 2.2pp

## 六、遗留问题

| 问题 | 现状 | 优先级 | 建议方案 |
|------|------|--------|---------|
| avg_watch_ratio 差 7% | 0.414 vs 0.445 | P2 | 提升 Band 2/3 Beta 参数 |
| like_rate 偏低 | 0.11% vs 0.48% | P2 | 调 boost 公式或提高 base_rate |
| warmup 期 30 交互 | 前 4% 未校准 | P3 | 可忽略或用先验初始化 CDF |
| AB 方向验证 | 未开始 | **P0** | 用 KuaiRand 随机 vs 正常推荐构造已知 AB |

## 七、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/agents/user_agent.py` | 修改 | 新增 quantile mapping、调 Beta 参数、调 skip 阈值 |
| `src/simulation/engine.py` | 修改 | run() 中重置 interest tracker |
| `scripts/validate_with_kuairand.py` | 修改 | 50 agent、temperature 0.2 |
| `docs/13-beta-calibration.md` | 修改 | 新增分位数映射章节 |
| `docs/14-50agent-validation-report.md` | 重写 | 完整校准前后对比报告 |
| `docs/15-calibration-session-log.md` | 新增 | 本文件 |

## 八、Git 提交记录

| Commit | 消息 |
|--------|------|
| `5ab17e5` | Beta distribution calibration: completion_rate 20.8% → 8.2% |
| `7bc5b15` | 50-agent validation report: completion aligned, variance needs work |
| `3e6e36f` | Quantile mapping calibration: skip_rate variance 19.5pp → 2.2pp |
