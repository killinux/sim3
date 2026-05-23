# AgentTorch 框架深入代码分析

> 仓库: https://github.com/AgentTorch/AgentTorch  
> 许可: AGPL-3.0（⚠️ 有商用限制，见下文）  
> 来源: MIT Media Lab  
> 规模验证: 单 GPU 840 万 agent

## 1. 核心架构

### 1.1 目录结构

```
agent_torch/
  core/
    runner.py            # nn.Module! 可微分模拟主循环
    controller.py        # Observe -> Act -> Progress 编排
    initializer.py       # 从 YAML 配置构建状态/子步骤
    registry.py          # 全局函数注册表
    substep.py           # 抽象基类 (Observation/Action/Transition)
    executor.py          # 高层便利运行器
    vectorized_runner.py # torch.vmap 向量化运行器
    distributed_runner.py # 多 GPU 数据并行
    helpers/
      soft.py            # 可微分逻辑: soft AND/OR/NOT, Gumbel-softmax
    distributions/
      distributions.py   # StraightThroughBernoulli, Categorical 等
    llm/
      archetype.py       # Archetype + LLMArchetype 类
      behavior.py        # Behavior 类 (prompt 分组 + 采样)
      template.py        # Template (可学习 Variable, P3O slots)
      Variable.py        # Variable 描述符 (可学习 prompt 表示优化)
  optim/
    p3o.py               # P3O 优化器 (REINFORCE on prompt)
  models/
    covid/               # SEIR-M 图传播 (37.5K-840万)
    macro_economics/     # 劳动/消费/金融市场 (270万)
    predator_prey/       # 空间捕食者-猎物 (120 agents)
```

### 1.2 关键设计决策

**Runner 继承自 `nn.Module`** —— 这是使整个模拟可微分的核心决策。所有子步骤中的可学习参数都注册在 PyTorch 计算图中。

## 2. 模拟循环

```python
# runner.py 的 _step_cpu_base (lines 121-153)
for time_step in range(num_steps):
    for substep in config["substeps"]:
        for agent_type in active_agents:
            observation = controller.observe(state, obs_fn, agent_type)
            action = controller.act(state, observation, policy_fn, agent_type)
        next_state = controller.progress(state, action, transition_fn)
        state = next_state
```

GPU 优化路径额外增加：
- 张量内存池（`_get_pooled_tensor` / `_return_to_pool`）
- CUDA 流管线化（异步 GPU→CPU 快照传输）
- 压缩快照（fp16 降精度, bool 打包）
- 活跃集检测（仅处理感染的 agent）

## 3. Agent 状态的张量表示

Agent 状态是按属性名索引的张量 dict。以 COVID 模型 37,518 agent 为例：

```python
state["agents"]["citizens"]["disease_stage"]   # shape: [37518, 1], int
state["agents"]["citizens"]["age"]             # shape: [37518, 1], int
state["agents"]["citizens"]["infected_time"]   # shape: [37518, 1], int
state["agents"]["citizens"]["is_quarantined"]  # shape: [37518, 1], bool
```

**GPU 加速原理**: 每个 agent 是张量的一行。"找到所有感染 agent" = `current_stages == EXPOSED_VAR`，一个向量化布尔掩码。**无逐 agent 的 Python 循环**。

所有属性张量通过 `_to_device()` 在初始化时移动到设备。CUDA 上使用 pinned memory + 轮询 CUDA streams。

## 4. 子步骤执行: Observation → Policy → Transition

三个抽象 `nn.Module` 子类：

```python
class SubstepObservation(nn.Module):
    def forward(self, state) -> dict

class SubstepAction(nn.Module):      # policy
    def forward(self, state, observation) -> dict

class SubstepTransition(nn.Module):
    def forward(self, state, action) -> dict
```

所有子步骤类将 YAML 参数解析为 `self.learnable_args`（`nn.ParameterDict`）和 `self.fixed_args`。当配置中 `calibration: true` 时，可学习参数额外存为 `calibrate_{name}` 属性，`requires_grad=True`。

### 注册与调度

```python
@Registry.register_substep("MakeIsolationDecision", "policy")
class MakeIsolationDecision(SubstepAction): ...
```

YAML 配置通过名称引用（如 `generator: NewTransmission`），Initializer 在 `registry.transition_helpers["NewTransmission"]` 中查找并实例化。

## 5. 可微分管线

### 5.1 自定义 Autograd 函数

```python
# StraightThroughBernoulli: 前向采样 Bernoulli，反向用恒等
class StraightThroughBernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        return torch.bernoulli(probs)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # 直通估计器

# Gumbel-Softmax 离散采样
def discrete_sample(probs, tau=0.1):
    return gumbel_softmax(probs.log(), tau=tau, hard=True)
```

### 5.2 软逻辑运算

```python
# soft.py
def compare(a, b):  # 可微分比较
    soft = sigmoid(hardness * (a-b))
    return hard + soft - soft.detach()  # 直通

def logical_and(a, b):  return a * b
def logical_or(a, b):   return a + b
```

### 5.3 梯度流路径（COVID 模型示例）

```
R (可学习参数)
  → _lam() 消息计算
  → new_transmission 聚合
  → exp(-transmission)
  → 概率
  → StraightThroughBernoulli
  → newly_exposed_today
  → daily_infected.sum()
  → loss
```

`StraightThroughBernoulli` 的直通确保 `d(loss)/d(R)` 有定义。

### 5.4 梯度校准循环

```python
runner = Runner(config, registry)
runner.init()
params = [p for p in runner.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-3)

for epoch in range(100):
    runner.reset_state()
    runner.step(num_steps)
    predicted = runner.state_trajectory[-1][-1]["environment"]["metrics"]
    loss = F.mse_loss(predicted, real_data)
    loss.backward()   # 梯度穿过整个模拟！
    optimizer.step()
    optimizer.zero_grad()
```

`CalibNN` 是一个编码器-解码器网络，从时间序列特征预测模拟参数，使用注意力序列 embedding + min/max 有界 sigmoid 输出。

## 6. LLM 集成: 原型方法

### 6.1 核心问题

不可能为 840 万 agent 各调用一次 LLM。

### 6.2 解决方案: 按属性分组

```
Template (prompt 模板 + 分组逻辑)
  → 按 grouping_logic 字段分组 agent
  → 每个唯一组合一次 LLM 调用
  → 广播结果到该组所有 agent
```

**具体流程** (`behavior.py`, `sample()` 方法):

1. Template 定义 prompt: `"You are {age} years old, {gender}, living in {region}..."`
2. `grouping_logic = ["age", "gender"]` → 相同 (age, gender) 的 agent 共享 prompt
3. `get_grouped_prompts()` 遍历人口，构建 profile dict，计算 `grouping_key()`（如 `"25|male"`），按 key 分桶
4. 每个唯一组一次 LLM 调用。6 个年龄组 × 2 个性别 = **仅 12 次 LLM 调用**（而非 37,518 次）
5. LLM 响应广播到该组所有 agent

```python
prompt_list, group_keys, group_indices = template.get_grouped_prompts(population)
for n_arch in range(archetype.n_arch):
    outputs = archetype[n_arch](prompt_list, last_k=12)
    for en, output in enumerate(outputs):
        idx = torch.tensor(group_indices[en], dtype=torch.long)
        sampled_behavior[idx, 0] += value_for_group
```

### 6.3 P3O: Prompt 表示优化

每个 `Variable` 有 5 种表示选项:
- 0: 跳过（省略字段）
- 1: 直接值（"25"）
- 2: 带标签（"age: 25"）
- 3: 上下文（"with 25"）
- 4: 描述性（"The age is 25"）

每个可学习 Variable 有一个 `nn.Parameter` logits 跨 5 个选项。P3O 使用 **REINFORCE + baseline** 优化哪种表示产生最佳 LLM 响应。这是**梯度优化的 prompt 工程**。

### 6.4 LLM 后端

- `DspyLLM`: DSPy + OpenAI，chain-of-thought，concurrent.futures 并行
- `MockLLM`: 返回 [low, high] 的随机浮点数，用于测试
- 可实现自定义后端: 实现 `LLMBackend.prompt(prompt_list)`

## 7. YAML 配置系统

使用 OmegaConf 做变量插值：

```yaml
simulation_metadata:
  num_agents: 37518
  device: auto
  calibration: true
  learning_params:
    lr: 0.001

state:
  agents:
    citizens:
      number: ${simulation_metadata.num_agents}
      properties:
        disease_stage:
          name: disease_stage
          shape: [${simulation_metadata.num_agents}, 1]
          dtype: int
          learnable: false
          value: 0

  network:
    agent_agent:
      type: network_from_file
      arguments:
        file_path: contact_network.pt

substeps:
  '0':  # Transmission
    active_agents: [citizens]
    observation: null
    policy:
      citizens:
        generator: MakeIsolationDecision
        arguments: {...}
    transition:
      generator: NewTransmission
      input_variables:
        disease_stage: agents/citizens/disease_stage  # 路径引用
      output_variables: [disease_stage, next_stage_time, ...]
```

`input_variables` 映射输出键到状态路径。Controller 用 `get_by_path` 读输入，`set_by_path` 写输出。

## 8. 现有模型分析

### 8.1 COVID（最复杂）

- **规模**: 37,518 agent（Astoria 社区），可扩展到 840 万（NYC）
- **子步骤**: 2 个/步（Transmission, Disease Progression），扩展版 4 个
- **Agent 属性**: 9 个（disease_stage, age, infected_time, is_quarantined 等）
- **网络**: 移动数据 contact network，GNN MessagePassing 做传播
- **可学习参数**: `R2`（每周繁殖数），`M`（死亡率）
- **LLM**: `MakeIsolationDecision` 用 LLM 原型决定隔离遵从

### 8.2 宏观经济

- **规模**: 2,712,360 agent（NYC 消费者）
- **子步骤**: 4 个/步（Earning, Consumption, Labor Market, Financial Market）
- **Agent 属性**: 14 个
- **LLM**: `WorkConsumptionPropensity` 融入经济上下文

## 9. 抖音适配设计

### 9.1 VideoRecommendation 模型结构

```yaml
state:
  agents:
    users:
      number: 1000000
      properties:
        interest_vector:          # [N, 64] 话题 embedding
        watch_history_embedding:  # [N, 128] 压缩历史
        fatigue_level:            # [N, 1] session 疲劳度
        engagement_score:         # [N, 1] 累积参与度
        session_time:             # [N, 1] 当前 session 秒数
        like_propensity:          # [N, 1] 点赞概率
        archetype_id:             # [N, 1] 1000 原型中的哪个

  objects:
    videos:
      properties:
        content_embedding: [V, 64]
        quality_score: [V, 1]
        creator_id: [V, 1]

substeps:
  '0':  # FeedGeneration - 推荐模型生成 feed
  '1':  # WatchDecision  - LLM 原型决定 观看/跳过/退出
  '2':  # EngagementAction - 点赞/评论/分享
  '3':  # InterestEvolution - 兴趣向量更新
```

### 9.2 实现 1000 用户原型

```python
class WatchDecisionTemplate(Template):
    src = "user_profiles.csv"
    grouping_logic = ["user_segment"]  # 按 archetype_id 分组

    prompt_string = """
    You are a {persona} user. Fatigue: {fatigue_level:.2f}.
    Video topic: {video_topic}.
    On 0-1, how likely to watch to completion?
    """

archetype = Archetype(prompt=WatchDecisionTemplate(),
                      llm=DspyLLM(...), n_arch=1)
# 1000 原型 → 1000 次 LLM 调用（而非 100 万次）
```

### 9.3 推荐模型作为子步骤

```python
@Registry.register_substep("generate_feed", "policy")
class GenerateFeed(SubstepAction):
    def __init__(self, ...):
        self.rec_model = TwoTowerModel(user_dim=64, item_dim=64)

    def forward(self, state, observation):
        user_emb = observation["interest_vector"]      # [N, 64]
        video_emb = state["objects"]["videos"]["content_embedding"]
        scores = torch.mm(user_emb, video_emb.T)       # [N, V]
        topk_scores, topk_idx = torch.topk(scores, k=10, dim=1)
        return {"feed_items": topk_idx, "feed_scores": topk_scores}
```

因为是 `nn.Module`，如果 `TwoTowerModel` 有可学习参数，**梯度会穿过推荐分数流入后续的观看/参与决策**。

### 9.4 离散动作的可微分处理

```python
# 二值 点赞/不点赞
like_prob = self.like_network(user_state, video_features)
like = StraightThroughBernoulli.apply(torch.stack([like_prob, 1-like_prob], -1))

# 多动作 (点赞/评论/分享/无)
action_logits = self.action_network(user_state)
action = Categorical.apply(torch.softmax(action_logits, -1))

# 观看时长 (连续，有界)
raw = self.duration_network(user_state, video_features)
watch_duration = torch.sigmoid(raw) * max_duration  # 可微分
```

## 10. 限制与挑战

### 10.1 内存

- **状态复制**: `controller.progress()` 中 `copy_module()` 每个子步骤深拷贝整个状态 dict
- 100 万 agent × 64 维 embedding = ~256MB/子步骤 × 4 子步骤 × 100 步 = **~100GB 分配/episode**
- 轨迹存储：400 个状态转换全部保存，可能耗尽 RAM

### 10.2 LLM 调用延迟

- 1000 原型 × 4 子步骤 × 100 步 = **40 万次 LLM 调用/episode**
- 即使 50ms/调用 + 批处理 = **5.5 小时/episode**
- 解决方案: 预计算所有 (原型, 上下文) 组合为查找表，或用本地小模型

### 10.3 可微分限制

- **LLM 调用本身不可微分**。梯度不能穿过原型采样路径
- P3O 通过 REINFORCE 优化 prompt 表示，不是通过 LLM 反向传播
- `copy_module` + `set_by_path` 模式脆弱：in-place 操作会破坏计算图

### 10.4 缺失功能

- ❌ 无内置物品侧动态（视频热度/趋势/内容衰减）
- ❌ 无注意力/transformer 子步骤
- ❌ 无流式/在线模式（固定长度 episode）
- ❌ 无动态 agent 创建/移除

### 10.5 ⚠️ AGPL-3.0 许可问题

`LICENSE.md` 确认 AGPL-3.0（尽管 `pyproject.toml` 声明 MIT，以 LICENSE.md 为准）。

**关键影响**:
- 任何修改或衍生作品必须同样 AGPL-3.0
- 如果部署为**网络服务**（如内部 AB 评估 API），整个衍生作品的源代码必须公开
- **对字节跳动/内部使用**: AGPL-3.0 通常与专有内部工具不兼容

**建议**: 
- (a) 获取 MIT Media Lab 的商业许可，或
- (b) 仅借鉴架构思想（不可版权保护），从零重写框架

## 11. 评估总结

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | ★★★★★ | 可微分 ABM + LLM 原型的组合非常优秀 |
| 缩放能力 | ★★★★☆ | 840 万 agent 已验证，但内存管理有改进空间 |
| 推荐场景适用 | ★★☆☆☆ | 无任何推荐系统组件，需全部自建 |
| 代码质量 | ★★★☆☆ | 研究代码，文档尚可，配置系统复杂 |
| 许可友好度 | ★☆☆☆☆ | AGPL-3.0 对商业使用有强限制 |
| 适配工作量 | 高 | 需自建推荐模型+视频场景全部子步骤 |

**核心价值**: 不一定直接使用代码，但其 **架构思想**（张量化状态 + 子步骤分解 + StraightThrough 梯度 + 原型分组）是最有价值的部分。
