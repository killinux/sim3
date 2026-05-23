# LLM Agent 用户模拟系统 - 调研总览

## 1. 项目目标

用 **1000 个 LLM Agent** 模拟抖音 **10 亿用户**的行为，在 AB 实验上线前验证线上效果。

**核心挑战**: 100 万倍的缩放差距（1K agent -> 1B user），要求：
- 模拟的行为分布尽可能贴近真实分布
- 能量化验证模拟与真实的近似程度
- 计算成本可控，支持快速迭代

## 2. 调研范围

本次调研覆盖了 2023-2026 年间 **30+ 篇论文**和 **10+ 个开源项目**，涵盖：

- LLM Agent 用户模拟（推荐系统方向）
- 大规模 Agent 模拟架构
- AB 实验的离线验证方法
- 模拟保真度验证方法论
- 工业界实践（Meta/Netflix/Google/Amazon/Shopify）
- 可用的公开数据集

## 3. 文档索引

| 文档 | 内容 |
|------|------|
| [02-papers-and-projects.md](02-papers-and-projects.md) | 论文与开源项目详细调研 |
| [03-approach-enumeration.md](03-approach-enumeration.md) | 14 种可能方案穷举与对比 |
| [04-scaling-and-validation.md](04-scaling-and-validation.md) | 缩放方法与验证方法论 |
| [05-implementation-patterns.md](05-implementation-patterns.md) | 实现架构与技术栈 |
| [06-recommended-approaches.md](06-recommended-approaches.md) | 推荐的 Top 3 组合方案 |
| [07-datasets.md](07-datasets.md) | 可用公开数据集 |
| [08-oasis-deep-dive.md](08-oasis-deep-dive.md) | OASIS 框架深入代码分析 |
| [09-agenttorch-deep-dive.md](09-agenttorch-deep-dive.md) | AgentTorch 框架深入代码分析 |
| [10-agent4rec-deep-dive.md](10-agent4rec-deep-dive.md) | Agent4Rec 框架深入代码分析 |
| [11-progress.md](11-progress.md) | **项目进度记录**（原理 + 进展 + 修正记录） |
| [12-kuairand-data-guide.md](12-kuairand-data-guide.md) | **KuaiRand 数据集使用指南**（数据结构 + 接入原理 + 验证方法） |

## 4. 关键结论速览

### 4.1 业界现状

- **最佳工业验证结果**: SimGym (Shopify, 2026) 达到 69% 符号一致性、0.64 Pearson 相关
- **最大规模**: AgentTorch (MIT) 单 GPU 模拟 840 万 agent
- **最直接相关**: AgentA/B (Amazon, 2025) 用 1000 agent 做 AB 测试验证
- **关键警示**: 开箱即用 LLM 的用户行为预测准确率仅 **11.86%**（Beyond Believability, 2025）

### 4.2 核心技术洞察

1. **原型聚类优于个体模拟** -- AgentTorch 证明聚合掉噪声后，原型方法在预测上反而更准
2. **验证应在分布级别，非个体级别** -- PAARS (Amazon) 的方法论
3. **必须微调** -- 开箱即用 LLM 准确率太低，fine-tune 后提升到 17.26%
4. **最优上下文: ~30 条近期行为** -- FineRob 的发现，多了反而加噪
5. **必须模拟反馈闭环** -- 推荐算法的自适应是 AB 测试的核心变量
6. **因果后校准可显著降低偏差** -- SYN-DIGITS 实现分布偏差降低 50-90%

### 4.3 推荐路径

对于暂无真实数据、处于调研阶段的情况，建议：

1. **用公开数据集（KuaiRand）做概念验证** -- 与抖音最接近的短视频数据
2. **以 OASIS 框架为基座改造** -- 已有社交媒体 + 推荐系统 + 百万级 agent 支持
3. **采用渐进式方案**: 方案A（混合多层架构）-> 方案B（可微分校准）-> 方案C（生态协同）
