# sim3 — LLM Agent 用户模拟系统

用 LLM Agent 模拟短视频平台用户行为，在 AB 实验上线前预测线上效果。

## 核心思路

```
真实世界                              模拟世界
─────────                            ─────────
10亿用户 → 推荐系统 → 看视频           50-1000个 LLM Agent → 代理推荐 → "看"视频
         → 点赞/跳过/评论                                 → 输出 interest_level
         → 收集指标                                       → Beta 分布采样 → watch_ratio
         → AB实验结论                                     → 预测 AB 结论
```

LLM 负责"理解"（这个视频对这个用户多有趣），统计模型负责"生成"（把理解映射到符合真实分布的行为）。

## 验证结果

### 分布对齐（vs 快手 KuaiRand 真实数据）

| 指标 | 模拟值 (50 agent, 5次均值) | 真实值 | 标准差 |
|------|--------------------------|--------|--------|
| skip_rate | 48.3% | 47.98% | ±2.2pp |
| completion_rate | 8.69% | 7.78% | ±1.2pp |
| avg_watch_ratio | 0.414 | 0.445 | ±0.015 |

### AB 方向验证

用 KuaiRand 随机曝光 vs 正常推荐构造已知方向 AB：

| 指标 | 模拟方向 | 真实方向 | 5次一致性 |
|------|---------|---------|----------|
| avg_watch_ratio | ↑ | ↑ | 5/5 |
| skip_rate | ↓ | ↓ | 5/5 |
| completion_rate | ↑ | ↑ | 5/5 |
| like_rate | ↑ | ↑ | 5/5 |

## 架构

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Persona 层  │    │   推荐层      │    │    决策层        │
│ 人口统计      │───→│ Surrogate    │───→│ LLM interest    │
│ 兴趣向量      │    │ RecSys       │    │ → 分位数映射校准  │
│ 行为签名      │    │ (epsilon-    │    │ → Beta 分布采样   │
│ 社交特征      │    │  greedy)     │    │ → watch_ratio    │
└─────────────┘    └──────────────┘    └─────────────────┘
                                              │
                                              ▼
                   ┌──────────────┐    ┌─────────────────┐
                   │   分析层      │◄───│    记忆层        │
                   │ Bootstrap CI  │    │ 工作记忆 30 条   │
                   │ BH 多重比较   │    │ 兴趣更新        │
                   │ A/A 验证      │    │ 疲劳累积        │
                   └──────────────┘    └─────────────────┘
```

## 快速开始

### 环境准备

```bash
pip install openai numpy scipy pandas pyyaml aiohttp
```

### 运行 KuaiRand 验证

```bash
# 需要 KuaiRand-Pure 数据集放在 data/KuaiRand-Pure/data/
DEEPSEEK_API_KEY="your-key" python3 scripts/validate_with_kuairand.py
```

### 运行 AB 方向验证

```bash
DEEPSEEK_API_KEY="your-key" python3 scripts/validate_ab_direction.py
```

### 运行模拟（合成数据）

```bash
DEEPSEEK_API_KEY="your-key" python3 scripts/run_simulation.py
```

## 项目结构

```
src/
  agents/
    persona.py          # 用户画像（人口统计+兴趣+行为签名）
    user_agent.py        # 核心 Agent（分位数映射+Beta 分布+参与行为）
    memory.py            # 三层记忆系统
  recommendation/
    content_pool.py      # 内容池（合成/真实视频）
    surrogate_recsys.py  # 代理推荐系统（embedding 匹配+epsilon-greedy）
  simulation/
    engine.py            # 模拟引擎（多 agent 并发+AB 分组）
    scheduler.py         # LLM 调度器（异步并发+限流）
  analysis/
    metrics.py           # 指标计算
    treatment_effect.py  # AB 效应分析（Bootstrap CI+BH 修正）
  validation/
    distribution_metrics.py  # 分布对比（Wasserstein/JS/KS）
    aa_test.py               # A/A 验证
  data/
    kuairand_loader.py   # 快手 KuaiRand 数据加载器

scripts/
  run_simulation.py          # 合成数据模拟
  validate_with_kuairand.py  # KuaiRand 分布对齐验证
  validate_ab_direction.py   # AB 方向一致性验证

docs/                        # 16 篇文档（调研+进度+校准+验证报告）
```

## 关键技术

### 混合模型

LLM 天然做二值决策（skip 或 watch_full），无法产生平滑的 watch_ratio 分布。解决方案：

1. LLM 输出 `interest_level` (1-10)
2. 分位数映射校准（强制分布、保留排序）
3. Beta 分布采样生成 `watch_ratio`
4. 参与动作（like/comment/share）由 persona base_rate 独立采样

### 分位数映射

LLM 的排序能力可靠（视频 A 比 B 有趣），但绝对值分布不稳定。分位数映射追踪 LLM 输出的经验 CDF，映射到预设的目标 CDF。无论 LLM 输出什么分布，最终都被标准化。skip_rate 标准差从 19.5pp 降到 2.2pp（9 倍改善）。

## 成本

| 场景 | Agent 数 | 耗时 | 成本 |
|------|---------|------|------|
| 单次验证 | 50 | ~46s | ~¥0.35 |
| 5 次均值验证 | 50×5 | ~4min | ~¥1.75 |
| 校准全过程（54 runs） | 20-50 | ~27min | ¥18.93 |

LLM: DeepSeek Chat (V3)，¥1/M input tokens, ¥2/M output tokens。

## 数据集

使用 [KuaiRand-Pure](https://github.com/chongminggao/KuaiRand)（快手，CC BY-SA 4.0）：27,285 用户 × 7,583 视频 × 118 万交互。包含随机曝光数据（无偏 ground truth）。

## 文档

详细技术文档见 `docs/` 目录，核心文档：

- [进度总览](docs/11-progress.md) — 完整项目状态和验证结果
- [Beta 校准原理](docs/13-beta-calibration.md) — Beta 参数和分位数映射的设计与迭代
- [50 Agent 验证报告](docs/14-50agent-validation-report.md) — 分布对齐详细报告
- [AB 方向验证](docs/16-ab-direction-validation.md) — AB 方向一致性验证报告
