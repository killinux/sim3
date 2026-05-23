# OASIS 框架深入代码分析

> 仓库: https://github.com/camel-ai/oasis  
> 许可: Apache 2.0  
> 规模: ~16,880 行 Python 代码  
> 状态: 活跃维护至 2026 年 5 月

## 1. 总体架构

OASIS 采用 **客户端-服务端模式**，通过异步消息队列中介：

```
┌──────────────────────┐        ┌──────────────────┐
│    SocialAgent(s)    │◄──────►│    Channel        │
│ (extends ChatAgent)  │ async  │ (asyncio Queue    │
│                      │ Queue  │  + Dict)          │
└──────────────────────┘        └────────┬───────────┘
                                         │
                                         ▼
                                ┌──────────────────┐
                                │    Platform       │
                                │ (SQLite backend)  │
                                │ 16 tables         │
                                └──────────────────┘
                                         │
                                         ▼
                                ┌──────────────────┐
                                │   OasisEnv        │
                                │ (orchestrator)    │
                                │ asyncio.gather()  │
                                └──────────────────┘
```

**关键架构弱点**: Platform 从单个队列 **顺序处理**所有动作，且 agent 用 `asyncio.sleep(0.1)` 轮询响应，每次往返增加 100ms+ 延迟。每个 agent 步骤需要 2-3 次 Channel 往返（refresh, group-listen, action）。

## 2. Agent 设计

### 2.1 Agent Prompt 结构

- **系统提示**: ~100-200 tokens 的 persona（名称、profile、平台身份）
- **观察提示**: JSON 序列化的推荐帖子 + 评论
- **工具定义**: ~500-800 tokens

**实测 Token 消耗**: ~3,524 tokens/agent/step（100 agent 时：335,600 input + 16,750 output）

**1000 agent × 10 步的预估**: ~35M tokens

### 2.2 动作系统

23+ 种动作类型，通过 `SocialAction` 类实现可扩展：

```
基础动作: create_post, like_post, unlike_post, dislike_post, undo_dislike_post
社交动作: follow_user, unfollow_user, mute_user, unmute_user
内容动作: create_comment, like_comment, unlike_comment, dislike_comment
群组动作: create_group, join_group, leave_group
高级动作: repost, quote_post, purchase_product
分析动作: do_nothing, create_poll, report_post, interview
```

**每个动作都是通过 LLM 的 tool calling 机制触发的**，不是硬编码的概率。

### 2.3 Agent 状态管理

Agent 状态跨步骤维护方式：
- **Profile**: 固定的 persona 描述（一次性设定）
- **记忆**: 由 CAMEL ChatAgent 的消息历史维护（无限增长，无压缩）
- **社交关系**: 存储在 SQLite 的 follow/mute 表中
- **发帖历史**: 存储在 SQLite 的 post/comment 表中

**问题**: 记忆无界增长，没有压缩或淘汰机制。

## 3. 推荐系统

四种内置算法：

| 算法 | 方法 | 个性化 | 性能 |
|------|------|--------|------|
| **RANDOM** | 随机采样 | 无 | 最快 |
| **REDDIT** | hot-score (时间衰减) | 无 | 快 |
| **TWITTER** | SentenceTransformer 余弦相似度 | 有（基于历史） | 很慢 |
| **TWHIN-BERT** | 批量编码 + 粗筛 4000 帖 + GPU 加速 | 有（基于 embedding） | 较快 |

**关键实现细节**:
- 所有算法将结果写入 `rec` 表
- 每步**完全删除**再**重新插入**所有推荐 —— 100K 行的全量操作
- TWHIN 对帖子做 embedding 编码后，与用户 profile embedding 做余弦相似度
- 推荐结果物化到缓存表 `rec`，通过 `update_rec_table()` 更新

## 4. 数据库设计

### 16 张 SQLite 表

```sql
-- 核心实体
user (user_id, name, bio, ...)
post (post_id, user_id, content TEXT, created_at, ...)  -- 仅文本！无视频
comment (comment_id, post_id, user_id, content, ...)

-- 社交关系
follow (follower_id, followee_id)
mute (muter_id, mutee_id)

-- 参与行为
like (user_id, post_id)
dislike (user_id, post_id)

-- 推荐
rec (user_id, post_id, ...)  -- 推荐结果缓存

-- 追踪
trace (user_id, action, post_id, timestamp, ...)  -- 行为日志
```

**关键缺失**:
- ❌ 无视频元数据（时长、封面、品类）
- ❌ 无观看行为追踪（观看时长、完播率、重播）
- ❌ 无曝光日志
- ❌ 无 AB 测试变体列
- ❌ `PRAGMA synchronous = OFF`（牺牲持久性换速度）
- ❌ `agent_environment.py` 为每个 agent 开**独立** SQLite 连接（导致锁竞争）

## 5. 并发与性能

### 核心并发模式

```python
# OasisEnv 主循环
async def step():
    semaphore = asyncio.Semaphore(128)  # 最大并发 LLM 调用
    tasks = [agent.step() for agent in agents]
    await asyncio.gather(*tasks)
```

### 1000 Agent 瓶颈分析

| 瓶颈 | 影响 | 估计耗时 |
|------|------|---------|
| SQLite 单写者 | 所有 DB 写入序列化 | ~2-3s/步 |
| Channel 轮询 | 100ms sleep × 3 往返/agent | ~300ms/agent |
| LLM API 调用 | 128 并发，~2s/调用 | ~16s/1000 agent |
| Rec 表重建 | DELETE ALL + INSERT 100K 行 | ~1-2s/步 |
| Agent 记忆 | 无界增长，无压缩 | 逐步增加 |

**预估每步总耗时**: ~20-26 秒（乐观估计，使用 GPU + 快速 LLM）

## 6. 代码质量评估

**优点**:
- 清晰的模块分离（`social_agent/`, `social_platform/`）
- 良好的异步模式设计
- 可扩展的动作系统（`SocialAction` 类）
- AgentGraph 支持（igraph/Neo4j）
- CAMEL 模型集成（多 LLM provider 支持）

**问题**:
- 生产代码中残留 `pdb.set_trace()` 调试代码（`recsys.py`）
- `recsys.py` 中模块级可变全局变量，非线程安全
- `TABLE_NAMES` set 中有 schema bug
- 多个 TODO 注释表明已知架构债务（特别是 1M agent 缩放时 `AgentGraph` 被放弃改用普通 list）
- 测试套件 31 个文件但大多需要 LLM API key 才能运行

## 7. 抖音适配改造分析

### 可复用部分（50-60%）

- ✅ Agent-Platform Channel 架构
- ✅ 动作分发模式
- ✅ Trace 日志系统
- ✅ AgentGraph（igraph/Neo4j）
- ✅ CAMEL 多模型集成
- ✅ Semaphore 并发控制
- ✅ ManualAction/LLMAction 区分

### 必须修改的部分

| 模块 | 当前 | 改造目标 |
|------|------|---------|
| Post 模型 | 纯文本 `content TEXT` | 视频元数据（时长/封面/品类/创作者） |
| 动作集 | quote_post, repost 等 | watch_video, scroll_past, replay, share_to_chat |
| 参与指标 | 二值 like/dislike | watch_time, completion_rate, 连续值 |
| 推荐算法 | 文本相似度 | 观看时长优化的 CF/two-tower |
| Channel 轮询 | `asyncio.sleep(0.1)` | `asyncio.Event`（零延迟通知） |
| Session/疲劳 | 无 | 疲劳模型 + session 生命周期 |
| AB 框架 | 无 | 变体配置 + 分组 + 指标收集 |
| 曝光追踪 | 无 | impression log + 倾向得分 |
| 数据库 | SQLite | PostgreSQL 或 DuckDB（并发写入） |

### 改造工作量估计

- **最小改造**（添加视频动作 + 基础推荐）: ~2-3 周
- **完整改造**（新 schema + 推荐 + AB + 验证）: ~6-8 周
- **建议**: 以 OASIS 为基座，逐步添加视频原生能力

## 8. 关键代码文件索引

| 文件 | 功能 | 改造优先级 |
|------|------|-----------|
| `social_agent/agent.py` | Agent 核心逻辑 + prompt | 高 |
| `social_platform/platform.py` | Platform 服务端 + 动作处理 | 高 |
| `social_platform/database.py` | SQLite schema + CRUD | 高 |
| `social_platform/recsys.py` | 推荐算法实现 | 高 |
| `social_agent/agent_action.py` | 动作定义（SocialAction 类） | 高 |
| `social_platform/channel.py` | 异步消息通道 | 中 |
| `oasis_env.py` | 模拟编排器 | 中 |
| `social_agent/agent_graph.py` | 社交图谱管理 | 低 |
