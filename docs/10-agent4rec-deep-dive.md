# Agent4Rec 框架深入代码分析

> 仓库: https://github.com/LehengTHU/Agent4Rec  
> 论文: SIGIR 2024, 清华大学  
> 规模: 1000 LLM Agent on MovieLens-1M  
> 成本: ~$12-16 / 全量运行  
> 时间: ~68 分钟 / 1000 agent

## 1. 目录结构

```
Agent4Rec/
  main.py                          # 入口: 创建 Arena 并运行
  parse.py                         # CLI 参数解析
  simulation/
    arena.py                       # Arena: 编排整个模拟循环
    avatar.py                      # Avatar: 单个 LLM Agent (Profile+Memory+Action)
    memory.py                      # AvatarMemory: LangChain 记忆 + 反思
    retriever.py                   # AvatarRetriver: FAISS 向量存储 + 时间衰减检索
    utils.py                       # 种子固定, precision/recall/F1
    vars.py                        # 全局可变状态 (token 计数, 线程锁)
  datasets/ml-1m/
    1_get_cf_data.py               # 步骤1: 采样1000用户, 创建 train/valid/test
    2_get_user_statistic.py        # 步骤2: 计算 活跃度/从众度/多样性 特征
    3_get_movie_detail.py          # 步骤3: 合并电影增强数据
    4_generate_persona.py          # 步骤4: LLM 生成品味 profile
    5_document_persona.py          # 步骤5: 解析 persona 为结构化 CSV
  recommenders/
    models/
      LightGCN.py, MF.py, MultVAE.py, Pop.py, Random.py, InfoNCE.py
```

## 2. 1000 Agent 初始化流程

### 2.1 离线 Pipeline（5 步）

**步骤 1**: 从 MovieLens-1M 的 6040 用户中随机采样 **1000** 个至少有 20 条正反馈（评分>3）的用户。按 40%/30%/30% 分 train/valid/test。

**步骤 2**: 计算三个社交特征（整数 1/2/3）:
- **Activity（活跃度）**: 总评分数，按 60th/90th 百分位分三档
- **Diversity（多样性）**: 覆盖 80% 观看量所需的不同流派数，按 33rd/66th 分三档
- **Conformity（从众度）**: 用户评分与电影平均评分的 MSE，按 25th/80th 分三档

**步骤 3**: 合并 GPT 生成的电影摘要与平均评分。

**步骤 4（关键）**: 用 `ThreadPoolExecutor(max_workers=1000)` **同时并发** 1000 用户的 persona 生成。

Persona 生成 Prompt:
```
System: I want you to act as a movie taste analyst roleplaying
        the user using first person "I".

User: Given a user's rating history:
      user gives high ratings for: <movies with ratings 4-5>
      user gives low ratings for: <movies with ratings 1-2>
      Generate TASTE-REASON pairs...
      Output format:
      TASTE: <descriptive taste>
      REASON: <brief reason>
```

**步骤 5**: 解析 1000 个 persona 文本文件为结构化 CSV (`all_personas_like_modify.csv`)，列: `[taste, reason, high_rating, low_rating]`

### 2.2 运行时初始化

```python
# arena.py
persona_df = pd.read_csv("all_personas_like_modify.csv")
user_statistic = pd.read_csv("user_statistic.csv")
for avatar_id in simulated_avatars_id:
    avatars[avatar_id] = Avatar(args, avatar_id,
        persona_df.loc[avatar_id],        # taste + high_rating
        user_statistic.loc[avatar_id])    # activity/conformity/diversity
```

每个 Avatar:
1. 解析 taste（管道符分隔 → 列表）
2. 将活跃度/从众度/多样性整数映射为**丰富自然语言描述**
3. 初始化 FAISS 记忆系统 + OpenAI embedding

## 3. Profile 模块深入

### 3.1 社交特征的自然语言映射

每个特征从整数 1/2/3 映射为 elaborate 的自然语言描述：

**Activity（活跃度）→ 控制退出倾向**:

| 级别 | 描述（英文原文摘要） | 行为效果 |
|------|-------------------|---------|
| 1 (低) | "极其难以捉摸的偶尔观众...哪怕一点点不满意就会立即退出" | 高退出概率 |
| 2 (中) | "偶尔观众...只对严格符合口味的电影好奇" | 中等退出概率 |
| 3 (高) | "电影狂热者...愿意观看几乎每部推荐电影" | 低退出概率 |

**Conformity（从众度）→ 控制评分行为**:

| 级别 | 描述 | 行为效果 |
|------|------|---------|
| 1 | "忠实跟随者...评分严重依赖历史评分" | 评分接近平均 |
| 2 | "平衡评估者...同时考虑历史评分和个人偏好" | 混合 |
| 3 | "特立独行者...完全忽略历史评分" | 独立评分 |

**Diversity（多样性）→ 控制探索倾向**:

| 级别 | 描述 | 行为效果 |
|------|------|---------|
| 1 | "极度挑剔的选择性观众" | 仅看匹配品味的 |
| 2 | "小众探索者...偶尔探索不同流派" | 有限探索 |
| 3 | "电影开拓者...不懈追求独特和冷门" | 广泛探索 |

### 3.2 注意：运行时 prompt 中没有人口统计特征

MovieLens 的人口统计数据（性别、年龄、职业）存储在 `agg_top_25.csv` 中，但在模拟 prompt 中**完全未使用**。只有品味描述和社交特征描述被注入 LLM prompt。

## 4. Memory 模块深入

### 4.1 架构

```
AvatarMemory (extends LangChain BaseMemory)
  │
  ├── memory_stream: list[Document]       # 内存中的文档流
  ├── vectorstore: FAISS                  # 向量检索
  │     └── OpenAIEmbeddings (ada-002, 1536维)
  └── reflection_threshold: 3             # 触发反思的阈值
```

### 4.2 写入机制

每次页面交互后存储 **两条** 记忆:

1. **事实记忆**: "推荐系统在第 N 页推荐了以下电影: {所有电影}，其中我观看了 {观看的电影} 并分别评分 {评分}。我不喜欢其余电影: {不喜欢的电影}。"

2. **决策记忆**: "浏览了 N 页后，我决定离开" 或 "翻到推荐第 N+1 页"

所有记忆 `importance_score = 1`（硬编码，无 LLM 重要性评分）。

### 4.3 检索机制

组合三个信号，通过 `_get_combined_score_list`:

```python
score = 0.9 * recency + 0.9 * importance + 1.0 * relevance
```

- **时效性**: `(1 - 0.01)^hours_passed`（但实际硬编码为 1，实质上禁用了）
- **重要性**: 来自元数据（始终为 1）
- **相关性**: FAISS 余弦相似度

Min-max 归一化后取 top-k（默认 k=5，某些路径用 k=10）。

### 4.4 反思机制

当 `aggregate_importance > 3`（即 3 次页面交互后）触发:

```
<交互历史记忆>

Given only the information above, describe your feeling of the
recommendation result using a sentence.
Output format:
[unsatisfied/satisfied] with the recommendation result because [reason].
```

反思结果作为新文档存回记忆，形成更高层级的综合记忆。

### 4.5 记忆容量

**无显式淘汰机制**。记忆无界增长。实际中每 agent ~5 页 = ~10-15 条记忆文档，不算多。但长模拟（如短视频的 100 次交互）会成为问题。

## 5. Action 模块深入

### 5.1 核心 Prompt（主决策）

**一次 LLM 调用产出三个结构化输出**:

```
System: You excel at role-playing. Picture yourself as a user exploring
a movie recommendation system. You have the following social traits:
Your activity trait is described as: {activity_dsc}
Your conformity trait is described as: {conformity_dsc}
Your diversity trait is described as: {diversity_dsc}
Beyond that, your movie tastes are: {taste joined by '; '}
And your rating tendency is {high_rating}
...
Relevant context from your memory: {retrieved_memories}

User: #### Recommended List ####
PAGE {current_page}
<- Movie Title -> <- History ratings: X.XX -> <- Summary: ... ->
...

1. 判断每部电影是否符合口味:
   MOVIE: [名]; ALIGN: [yes/no]; REASON: [原因]

2. 从符合口味的电影中选择要看的:
   NUM: [数量]; WATCH: [所有选择的电影名]; REASON: [原因]

3. 为选择观看的电影评分(1-5):
   MOVIE: [电影名]; RATING: [1-5]; FEELING: [感受]
```

**关键设计**: 结构本身就是思维链（align → select → rate），无需显式 CoT 请求。

### 5.2 退出决策（独立 LLM 调用）

```
System: ...activity trait... Now you are in Page {current_page}.
You may get tired with the increase of pages browsed.
(above 2 pages is a little bit tired, above 4 pages is very tired)
Relevant context from your memory: {retrieved_memories}

User: Generate overall feeling based on memory + activity trait.
POSITIVE: [reason] / NEGATIVE: [reason]
Assess fatigue level...
Decide continue or exit...
To leave: [EXIT]; Reason: [reason]
To continue: [NEXT]; Reason: [reason]
```

**无显式退出概率模型** —— 退出完全委托给 LLM 基于活跃度特征、疲劳提示和记忆上下文的判断。

### 5.3 每页每 agent 的 LLM 调用次数

| 调用 | 目的 | Tokens 估算 |
|------|------|------------|
| 1 | `reaction_to_recommended_items()` | ~2000 input, ~500 output |
| 2 | 记忆反思（如触发） | ~500 input, ~200 output |
| 3 | `make_next_decision()` | ~1500 input, ~200 output |
| 4 | 退出后 interview | ~500 input, ~200 output (一次性) |

## 6. 模拟循环

### 6.1 预计算排序

```python
# abstract_arena.py
def get_full_rankings():
    # 预计算所有用户的所有物品排序
    # 创建完整 n_users × n_items 分数矩阵，argsort
    # 训练/测试物品排除（设为 -inf）
```

**推荐是一次性预计算的，不是在线更新的。** 这是一个关键限制。

### 6.2 并行执行

```python
# arena.py
loop = asyncio.get_event_loop()
executor = ThreadPoolExecutor(max_workers=500)
tasks = [self.async_simulate_one_avatar(id, loop, executor)
         for id in simulated_avatars_id]
loop.run_until_complete(asyncio.wait(tasks))
```

1000 agent 同时启动，最多 500 线程。瓶颈是 OpenAI API 速率限制。

### 6.3 Agent 循环

```python
while not avatar.exit_flag:
    id_on_page = next(page_generator)  # 获取下 4 个物品
    movies_on_page = [movie_detail.loc[idx] for idx in id_on_page]
    response = avatar.reaction_to_recommended_items(formatted_items, page_num)
    # 正则解析响应
    # 记录行为 (watch/rate/align/like)
    # 检查最大页数
```

### 6.4 错误处理

指数退避 + 抖动:
```python
except_waiting_time = 1
max_waiting_time = 16
while response == '':
    try: # API 调用
    except: 
        time.sleep(random.randint(0, except_waiting_time-1))
        except_waiting_time = min(except_waiting_time * 2, max_waiting_time)
```

无限重试直到成功，无熔断器。

## 7. 保真度评估

### 7.1 评分分布

Agent 的评分分布与真实 MovieLens 匹配（峰值在 4 分），但关键失败：**无法生成 1-2 分低评分**（LLM 先验知识 + 礼貌偏差）。

### 7.2 验证模式

混合已知正样本 + 随机未观测样本，测量 agent 观看决策的：
- 平均 Recall: **50.28%**
- 平均 Precision: **41.55%**

### 7.3 社交特征影响

ANOVA F-test 验证不同社交特征组产生**统计显著的不同行为**。

### 7.4 幻觉处理

**无显式处理**。依赖提示中的 "stay grounded in reality" 指令和结构化输出解析。LLM 生成不存在的电影名时，`title_id_dict` 查找静默失败。

## 8. 实际运行数据

| 指标 | 值 |
|------|---|
| 总成本（1000 agent） | **$12.43** |
| 总 token | 690 万 |
| 每 agent 成本 | ~$0.012-0.016 |
| 总耗时 | **67.8 分钟** |
| 最快 agent | 16.3 秒（1 页就退出） |
| 最慢 agent | 4055.6 秒（5 页全看完） |
| 平均退出页 | 3.234 |
| 模型 | gpt-3.5-turbo |
| Temperature | 0.2 |

## 9. 抖音适配分析

### 9.1 可直接复用的部分

- ✅ 社交特征分层系统（activity/conformity/diversity）—— 领域无关
- ✅ FAISS 记忆 + 检索架构
- ✅ 异步并行执行模式（500 线程，1000 agent）
- ✅ Prompt 模板结构（system context + user instruction）
- ✅ 指数退避错误处理

### 9.2 必须修改的部分

**a) 数据表示**: MovieLens `(title, genres, rating, summary)` → 视频 `(video_id, creator, tags, duration, description, like_count, completion_rate)`

**b) 动作空间**: 从"评价一批电影"到"逐条视频反应"

当前（每页 4 个电影批量评估）:
```
MOVIE: [名]; ALIGN: [yes/no]; REASON: [...]
NUM: [N]; WATCH: [选择的电影]; REASON: [...]
MOVIE: [名]; RATING: [1-5]; FEELING: [...]
```

改为（逐条视频顺序评估）:
```
You are scrolling through a short video feed. Next video:
[VIDEO]: {creator} - {description} ({duration}s)
[TAGS]: {tags}
[STATS]: {likes}K likes, {comments}K comments

1. DECISION: [watch_full / watch_partial_{seconds}s / skip]
2. If watched: ENGAGE: like=[yes/no]; comment=[yes/no];
   share=[yes/no]; follow=[yes/no]
3. FEELING: [brief aftermath]
4. ENERGY: [continue_scrolling / take_break / close_app]
```

**c) 推荐集成**: 从一次性预计算 → 每次交互后在线更新

**d) 社交图谱**: Agent4Rec **无社交图** —— 每个 agent 独立操作。抖音需要关注关系。

**e) LLM 调用量**: 从每页 1 次（4 个物品）→ 每视频 1 次。约 **翻倍**调用量。但每次调用更简单更便宜。

**f) 成本估算**: 以 gpt-3.5-turbo 计，每 session ~100 视频 → 约 $0.05-0.10/agent/session。1000 agent = $50-100/模拟天。用 Qwen-7B 自部署可降低 10-50x。

## 10. 关键改造文件

| 文件 | 改造内容 | 优先级 |
|------|---------|--------|
| `simulation/avatar.py` | 所有 prompt 模板、动作空间、profile 解析 | 最高 |
| `simulation/arena.py` | 模拟循环、物品展示、响应解析 | 最高 |
| `simulation/memory.py` | 记忆内容格式、反思 prompt | 高 |
| `datasets/*/4_generate_persona.py` | 视频用户 persona 生成 prompt | 高 |
| `datasets/*/2_get_user_statistic.py` | 从视频参与数据计算社交特征 | 高 |

## 11. 评估总结

| 维度 | 评分 | 说明 |
|------|------|------|
| 1000 Agent 验证 | ★★★★★ | 唯一在 1000 agent 规模验证的推荐模拟 |
| 认知架构 | ★★★★☆ | Profile+Memory+Action 三模块设计清晰 |
| 成本效率 | ★★★★★ | $12.43 / 1000 agent 非常便宜 |
| 保真度 | ★★★☆☆ | 评分分布大致匹配但无法产生低评分 |
| 可扩展性 | ★★☆☆☆ | MovieLens 强耦合，改造工作量中等 |
| 代码质量 | ★★★☆☆ | 研究代码，可读性好但无测试 |
| 抖音适配难度 | 中 | 核心架构可复用，动作空间需重设计 |
