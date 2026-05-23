# 论文与开源项目详细调研

## 1. 基础架构类

### 1.1 Generative Agents / Stanford Smallville (2023 UIST)

- **作者**: Park et al., Stanford + Google
- **方法**: 25 个 agent，每个有 Memory Stream（记忆流）+ Reflection（反思）+ Planning（计划）三个核心模块
- **意义**: 所有后续 LLM Agent 模拟工作的基础架构
- **代码**: https://github.com/joonspk-research/generative_agents
- **局限**: 规模仅 25 agent，无推荐系统，无量化验证

### 1.2 Agent4Rec (2024 SIGIR)

- **作者**: Zhang et al., 清华大学
- **方法**: 在 MovieLens-1M 上构建 **1000 个 LLM Agent**，每个有三模块：
  - **Profile 模块**: 社交特征量化（活跃度/从众度/多样性）+ GPT 生成 25 项兴趣摘要
  - **Memory 模块**: 双格式（事实记忆 + 情感记忆），支持检索/写入/反思操作
  - **Action 模块**: 品味驱动（观看/评分）+ 情感驱动（退出/访谈，使用 CoT 推理）
- **规模**: 1000 agent，GPT-3.5-Turbo，总成本约 $16
- **保真度评估**:
  - Agent 的评分分布与 MovieLens 真实分布匹配（峰值都在 4 分）
  - 但关键失败：**无法生成 1-2 分的低评分**（LLM 先验知识干扰）
  - 精度随负样本比例增加急剧下降（1:9 时从 ~70% 降到 ~25%）
  - 不同社交特征层级的 agent 行为有统计显著差异
- **代码**: https://github.com/LehengTHU/Agent4Rec
- **抖音适用性**: 中低。好的认知架构模板，但只支持电影评分，无视频信号（观看时长、滑动、重播），无社交图谱，幻觉问题未解决

### 1.3 RecAgent / YuLan-Rec (2025 ACM TOIS)

- **作者**: Wang et al., 人民大学
- **方法**: Agent 之间有社交互动 + 与推荐系统交互，支持反事实分析
- **代码**: https://github.com/RUC-GSAI/YuLan-Rec
- **抖音适用性**: 中等。有社交维度，但规模受限

### 1.4 InteRecAgent / RecAI (2024 WWW)

- **作者**: Microsoft Research
- **方法**: LLM 作为"大脑"，传统推荐模型作为"工具"
- **代码**: https://github.com/microsoft/RecAI
- **抖音适用性**: 中等。LLM + 传统模型的混合思路有参考价值

### 1.5 Concordia (2024 NeurIPS)

- **作者**: Google DeepMind
- **方法**: 生产级 GABM 框架，RPG Game Master 模式
- **代码**: https://github.com/google-deepmind/concordia
- **抖音适用性**: 低。通用框架，需大量定制

### 1.6 LLM-Powered User Simulator (2025 AAAI)

- **作者**: CityU HK + 快手
- **方法**: 逻辑模型（LLM 提取特征/蒸馏情感/应用偏好规则）+ 统计模型（GBT/CF）的集成
- **关键结果**: 集成可训练 RL 推荐策略（A2C/DQN/PPO/TRPO），证明合成数据可用
- **代码**: https://github.com/Applied-Machine-Learning-Lab/LLM_User_Simulator

---

## 2. 大规模模拟架构类

### 2.1 AgentTorch (2025 AAMAS, MIT Media Lab)

- **核心突破**: "LLM 原型"方法 —— 将数百万用户聚类为代表性行为原型，**LLM 每原型仅调用一次**，结果应用到所有匹配 agent
- **规模**: 单 GPU 模拟 **840 万 agent**（NYC COVID 场景）
- **架构**:
  - 运行时 = 配置（YAML + 可学习属性）+ 注册表（装饰器子步骤类）+ Runner/Controller
  - 每个子步骤是 `torch.nn.ModuleDict`，包含 Observation -> Policy -> Transition 三阶段
  - 全部可微分 PyTorch 模块，支持反向传播
- **梯度校准**: 比黑盒方法快 **8300x**。通过自动微分穿过随机动力学，替代暴力参数搜索
- **关键洞察**: **原型方法在预测上优于个体模拟** —— 因为个体 LLM 调用引入噪声，原型聚合掉噪声后捕捉结构化模式
- **代码**: https://github.com/AgentTorch/AgentTorch (AGPL-3.0)
- **抖音适用性**: 中高。已证明规模、可微分、GPU 原生、可扩展。但无推荐系统模型，原型丢失个体丰富度
- **改造方向**: 定义用户 agent（interest_vector, watch_history, fatigue, social_connections）、内容对象（topic_vector, duration, engagement_counts）、子步骤（FeedGeneration, WatchDecision, EngagementUpdate）

### 2.2 OpenCity (2025 ACL, 清华大学)

- **核心方法**: Group-and-Distill prompt 优化
- **规模**: **1 万 agent** 在 1 小时内完成一天的日常活动模拟
- **优化效果**: 每 agent **600x 加速**，LLM 请求减少 **73.7%**，token 减少 **45.5%**
- **验证**: 与 6 个真实城市数据对比验证
- **抖音适用性**: 中等。优化技术可迁移

### 2.3 GenSim (2025 NAACL)

- **核心方法**: 纠错机制 + 长期模拟
- **规模**: 支持 **10 万 agent**
- **代码**: https://github.com/TangJiakai/GenSim

### 2.4 SocioVerse (2025, 复旦大学)

- **核心方法**: 从真实社交媒体数据构建 **千万级用户池**
  - 101 万来自 X/Twitter + 916 万来自小红书
  - 15 个人口统计维度，LLM 标注 + 人工验证（一致性 0.849-0.956）
- **四层对齐**: 社会环境 -> 用户引擎(IPF/IDS 采样) -> 场景引擎(4 模板) -> 行为引擎(LLM+ABM 混合)
- **验证结果**:
  - 总统选举预测准确率 **92.2%**（Qwen2.5-72b）
  - 新闻反馈 KL 散度 0.196（GPT-4o）
  - 经济调查 NRMSE 0.023-0.036
  - 所有模型都表现出保守偏差
- **代码**: https://github.com/FudanDISC/SocioVerse （核心模拟代码未发布，仅问卷和评估脚本）
- **抖音适用性**: 中等。大规模中国用户池有价值，但侧重调查场景非信息流

### 2.5 OASIS (2024, CAMEL-AI)

- **核心方法**: 社交媒体数字孪生
- **规模**: 支持 **100 万 agent**
- **架构**:
  - AsyncIO 并发执行
  - SQLite 后端（user/post/like/dislike/follow/mute/rec/trace 表）
  - 基于频道的消息传递
  - LLM + 规则 agent 混合
  - **23+ 动作类型**: post/comment/follow/mute/group/e-commerce/interview
- **内置推荐算法**: RANDOM, TWITTER(基于历史), TWHIN(基于embedding), REDDIT(热度时效)
- **Token 成本**: 100 agent / 1 时间步 / Qwen Turbo = ~336K input + ~17K output tokens
- **代码质量**: 好。清晰的分离（social_agent/, social_platform/），良好的异步模式，可扩展的动作系统。Apache 2.0 许可
- **抖音适用性**: **高**。已有社交媒体 + 内置推荐系统 + 社交图谱 + 23+ 交互类型
- **需要补充的能力**: 视频原生动作（watch_time/completion_rate/replay）、多模态内容感知、更强的推荐算法

---

## 3. AB 实验验证类

### 3.1 AgentA/B (2025, Amazon + Northeastern + Penn State)

- **arXiv**: 2504.09723
- **核心方法**:
  - LLM 生成 **10 万个用户 persona**（人口统计 + 偏好 + 购物目标）
  - 抽样 **1000 agent** 做 AB 测试（500 treatment / 500 control）
  - 使用 **Claude 3.5 Sonnet**
  - 16 个计算节点，Selenium 无头浏览器，每 session 20 个动作
  - 三模块: 环境解析(HTML->JSON) + LLM Agent + 动作执行(Selenium+容错)
- **验证结果**:
  - 与真实 Amazon AB 测试（200 万人类用户）方向一致
  - Treatment 组显著更多购买行为（chi-squared=5.51, p<0.05）
  - Agent 比人类更"目标导向"（6 vs 16 个动作）
  - 子群体模式自然涌现（老年用户更受益于简化过滤器）
- **代码**: 无公开仓库
- **抖音适用性**: 中等。AB 测试方法论好，但无代码，网页浏览为主非信息流

### 3.2 PAARS (2025, Amazon, ACL REALM)

- **arXiv**: 2503.24228
- **核心方法**: 从真实购物数据挖掘 persona，**在群体/分布级别验证**（非个体级别）
- **关键洞察**: KL 散度衡量分布距离，**2/3 方向一致性**
- **抖音适用性**: 高。分布级验证方法论直接可用

### 3.3 SimGym (2026, Shopify)

- **核心方法**: 浏览器级 agent，多模态 LLM 与真实 UI 交互
- **规模**: 2000 并发浏览器 agent，48 B200 GPU，40 万 sessions/天
- **验证结果**: **69% 符号一致性**，**0.64 Pearson 相关** —— 当前工业界最佳
- **抖音适用性**: 中低。最高保真度但成本极高，视频内容成本更甚

### 3.4 A/B Agent (2026 KDD, CityU HK)

- **arXiv**: dl.acm.org/doi/10.1145/3770854.3785688
- **核心方法**: 多模态沙箱 + **疲劳系统**（信息过载建模）
- **代码**: https://github.com/Applied-Machine-Learning-Lab/ABAgent
- **抖音适用性**: 高。专门针对短视频场景，有疲劳模型

### 3.5 SYN-DIGITS (2026)

- **核心方法**: 因果后校准层 —— 学习 LLM 模拟的系统性偏差函数，对新模拟做分布级修正
- **关键结果**: 个体相关提升 50%，分布偏差降低 **50-90%**
- **抖音适用性**: 高。模型无关，可叠加到任意模拟方案上

---

## 4. 保真度验证 — 关键警示论文

### 4.1 Beyond Believability (2025, Amazon/Northeastern)

- **arXiv**: 2503.20749
- **关键发现**: 开箱即用 LLM 的用户动作生成准确率仅 **11.86%**；fine-tuning 后提升到 **17.26%**
- **启示**: "看起来合理"和"分布准确"之间有巨大鸿沟

### 4.2 Lost in Simulation (2026)

- **arXiv**: 2601.17087
- **关键发现**: LLM 模拟用户是不可靠的代理
  - 跨 LLM 成功率变异 **9 个百分点**
  - 系统性校准失败
  - 人口统计差异会复合放大

### 4.3 FineRob (2024)

- **arXiv**: 2412.03148
- **关键发现**:
  - **~30 条近期行为**是最优上下文长度，更多反而引入噪声
  - 仅用用户 profile 价值有限
  - OM-CoT fine-tuning 可提升 F1 **4.5-9.8%**

### 4.4 OmniBehavior (2026, 中科院 + 快手)

- **关键发现**: 冻结 LLM 模拟器的三大系统偏差
  - **评分中心化**: RLHF "礼貌偏差"导致评分偏向中高
  - **曝光偏差**: Agent 过度服从推荐
  - **动作简化**: Agent 缺乏真实人类的犹豫/放弃模式

---

## 5. 相关领域重要论文

### 5.1 用户行为微调

| 论文 | 方法 | 关键结果 |
|------|------|---------|
| **UserMirrorer** (ACL 2026) | SFT+DPO 微调 7B 模型 | 微调 7B 优于冻结 GPT-4，8 个领域验证 |
| **PersonaTwin** (ACL GEM 2025) | 多层 prompt conditioning | 单个基座模型 8500+ 个体数字孪生 |
| **PersonaAct** (2026) | 从抖音/快手/B站真实数据自动合成 persona | SFT + RL 微调，多模态观察 |

### 5.2 推荐协同模拟

| 论文 | 方法 | 关键结果 |
|------|------|---------|
| **FLOW** (2024) | 用户-推荐反馈闭环协同模拟 | 捕捉"推荐塑造偏好"动态 |
| **RecInter** (EMNLP 2025) | 用户-推荐交互模拟 | 复现品牌忠诚度和马太效应 |
| **TriRec** (2026) | 三方架构 (User+Item+Platform) | 同时提升准确率和公平性 |
| **RecoWorld** (Meta, WWW 2026) | 双视角推荐世界模型 | 多轮交互 + 反思指令 |
| **CreAgent** (2025) | 创作者 agent + 博弈论信念机制 | 长期创作者行为模拟 |

### 5.3 短视频专项

| 论文 | 方法 | 关键结果 |
|------|------|---------|
| **SimTok** (2025) | 短视频 filter bubble 模拟 | 使用抖音类特征（年龄/性别/城市/设备） |
| **LLM-Augmented Digital Twin** (2026.03) | 四孪生架构 (User+Content+Interaction+Platform) | 专门针对短视频平台策略评估 |
| **SimUSER** (2025) | Persona匹配+视觉感知+记忆+决策 | 四认知模块用户模拟器 |

---

## 6. 工业界实践

### 6.1 Meta
- "Test Universe" 模拟用户群用于 Facebook/Instagram 测试（代码覆盖率提升 38%）
- RecoWorld: 推荐世界模型，多轮用户-推荐交互

### 6.2 Netflix
- 离线 OPE（Doubly Robust 估计器）
- 异质性处理效应分析
- Slate-level 离策略评估

### 6.3 Google
- RecSim / RecSim NG: 参数化用户模型框架，基于 Edward2+TensorFlow
- 支持自动微分、MCMC 推断、EM/GAN 训练
- **注意**: 已于 2026 年 4 月归档(ARCHIVED)，不再维护

### 6.4 Shopify
- SimGym: 浏览器级模拟，48 B200 GPU
- 40 万 sessions/天
- 当前工业界最佳验证结果（0.64 Pearson）

### 6.5 Amazon
- AgentA/B: 1000 agent AB 测试
- PAARS: 分布级别验证方法论
- 使用 Claude 3.5 Sonnet

### 6.6 字节跳动/抖音
- 未发现公开的模拟/合成用户方案
- 快手有参与 LLM User Simulator (AAAI 2025) 和 OmniBehavior (2026)
