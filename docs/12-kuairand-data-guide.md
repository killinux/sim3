# KuaiRand 数据集使用指南

## 一、数据集简介

KuaiRand 是快手（Kuaishou）发布的短视频推荐数据集，发表于 CIKM 2022。它最大的特点是包含**随机曝光数据**——在正常推荐流中随机插入视频，消除了推荐算法的选择偏差，是目前短视频领域最接近"无偏 ground truth"的公开数据。

| 属性 | 值 |
|------|---|
| 来源 | 快手（与抖音同为短视频平台，用户行为模式最接近） |
| 许可 | CC BY-SA 4.0（可商用，需署名+相同方式共享） |
| 论文 | KuaiRand: An Unbiased Sequential Recommendation Dataset (CIKM 2022) |
| 代码 | https://github.com/chongminggao/KuaiRand |

### 三个子集

| 子集 | 用户 | 视频 | 交互 | 大小 | 用途 |
|------|------|------|------|------|------|
| **KuaiRand-Pure** | 27,285 | 7,583 | 118 万 | 194MB | **原型验证（当前使用）** |
| KuaiRand-1K | 1,000 | 440 万 | 1,170 万 | 4.3GB | 1000 用户深度序列建模 |
| KuaiRand-27K | 27,285 | 3,200 万 | 3.22 亿 | 46GB | 全量数据，RL/OPE 评估 |

我们当前使用 **KuaiRand-Pure**（最小子集），保留了随机曝光候选池中 7,583 个视频的所有交互。

---

## 二、数据里有什么

### 2.1 交互日志（核心）

每条记录 = 一个用户看了一个视频后的完整反馈。

**文件**: `log_random_4_22_to_5_08_pure.csv`（随机曝光）、`log_standard_*.csv`（正常推荐）

| 字段 | 类型 | 含义 | 示例 |
|------|------|------|------|
| `user_id` | int | 用户 ID | 0 |
| `video_id` | int | 视频 ID | 5543 |
| `date` | int | 日期 YYYYMMDD | 20220430 |
| `hourmin` | int | 时分 HHMM | 1800 |
| `time_ms` | int | 毫秒时间戳 | 1651314030792 |
| **12 种反馈信号** | | | |
| `is_click` | 0/1 | 是否点击（双列 UI）或有效播放（单列 UI） | 0 |
| `is_like` | 0/1 | 是否点赞 | 0 |
| `is_follow` | 0/1 | 是否关注创作者 | 0 |
| `is_comment` | 0/1 | 是否评论 | 0 |
| `is_forward` | 0/1 | 是否转发/分享 | 0 |
| `is_hate` | 0/1 | 是否踩 | 0 |
| `long_view` | 0/1 | 是否长时间观看 | 0 |
| `play_time_ms` | int | **实际观看时长（毫秒）** | 863 |
| `duration_ms` | int | **视频总时长（毫秒）** | 30066 |
| `profile_stay_time` | int | 在创作者主页停留时间 | 0 |
| `comment_stay_time` | int | 在评论区停留时间 | 0 |
| `is_profile_enter` | 0/1 | 是否进入创作者主页 | 0 |
| **元数据** | | | |
| `is_rand` | 0/1 | **是否随机曝光**（1=随机插入，0=正常推荐） | 1 |
| `tab` | int | 推荐场景 [0-14]（主 feed、发现页等） | 1 |

**关键字段解读**：

- `play_time_ms / duration_ms` = **观看比例（watch_ratio）**，这是最重要的隐式反馈
- `is_rand=1` 的数据是**无偏**的——视频是随机给用户看的，不是推荐算法选的
- 12 种反馈信号覆盖了短视频的全部交互类型

### 2.2 用户特征

**文件**: `user_features_pure.csv`，31 个字段

| 字段 | 含义 | 取值示例 |
|------|------|---------|
| `user_active_degree` | 活跃度 | full_active / high_active / middle_active / low_active |
| `is_live_streamer` | 是否主播 | 0/1 |
| `is_video_author` | 是否创作者 | 0/1 |
| `follow_user_num` | 关注数 | 具体数字 |
| `fans_user_num` | 粉丝数 | 具体数字 |
| `friend_user_num` | 好友数 | 具体数字 |
| `register_days` | 注册天数 | 具体数字 |
| `follow_user_num_range` | 关注数分档 | 0 / (0,10] / (10,50] / ... / 500+ |
| `fans_user_num_range` | 粉丝数分档 | 0 / [1,10) / [10,100) / ... |
| `onehot_feat0~17` | 加密特征（人口统计） | 整数编码 |

**注意**: 没有明文的年龄/性别/城市，但 `onehot_feat0~17` 是加密后的人口统计特征（快手为保护隐私做了脱敏）。

### 2.3 视频基础特征

**文件**: `video_features_basic_pure.csv`，12 个字段

| 字段 | 含义 | 示例 |
|------|------|------|
| `video_id` | 视频 ID | 5543 |
| `author_id` | 创作者 ID | 123456 |
| `video_type` | 类型 | NORMAL / AD |
| `upload_dt` | 上传日期 | 2022-03-15 |
| `video_duration` | **时长（毫秒）** | 30066 |
| `server_width/height` | 分辨率 | 720 × 1280 |
| `music_id` | 背景音乐 ID | 98765 |
| `tag` | **品类标签（逗号分隔）** | "搞笑,段子" |

### 2.4 视频统计特征

**文件**: `video_features_statistic_pure.csv`，50+ 个字段

每个视频一个月内的**日均**统计量（浮点数）：

| 类型 | 字段 |
|------|------|
| 曝光/播放 | `show_cnt`, `play_cnt`, `complete_play_cnt`, `valid_play_cnt`, `play_progress` |
| 参与 | `like_cnt`, `comment_cnt`, `follow_cnt`, `share_cnt`, `download_cnt`, `collect_cnt` |
| 质量 | `report_cnt`, `reduce_similar_cnt` |

`play_progress` = 平均播放进度，是衡量视频质量的好指标。

---

## 三、为什么随机曝光数据特别有价值

正常推荐系统的数据有**选择偏差**：

```
正常推荐数据（is_rand=0）:
  推荐算法觉得用户喜欢 → 才推荐 → 用户看了
  问题: 不喜欢的内容根本不会被推荐，所以看不到负面数据
  
随机曝光数据（is_rand=1）:
  随机选一个视频 → 强制插入 feed → 用户看了
  优势: 无论用户喜不喜欢，都有数据，分布是无偏的
```

**这对我们的模拟验证意义重大**：

用随机曝光数据做 ground truth 验证，结果不受推荐算法偏差影响。如果模拟的 watch_ratio 分布能逼近随机曝光数据的分布，说明 Agent 的行为模式确实在模拟真实用户的偏好，而不是在模拟推荐算法的偏好。

---

## 四、数据如何接入模拟系统

### 4.1 整体流程

```
KuaiRand 数据
    │
    ├──→ ① 构建 Persona（真实用户画像）
    │     user_features.csv 的活跃度/粉丝数
    │     + 交互日志计算的 like_rate/skip_rate/兴趣分布
    │     → 替代随机生成的 Persona
    │
    ├──→ ② 构建内容池（真实视频）
    │     video_features_basic.csv 的时长/品类/创作者
    │     + video_features_statistic.csv 的播放量/点赞量
    │     → 替代合成视频
    │
    └──→ ③ 提供 Ground Truth（验证基准）
          交互日志的 play_time_ms/is_like/is_comment
          → 计算真实分布
          → 模拟分布 vs 真实分布 → Wasserstein/JS 距离
```

### 4.2 Persona 构建细节

从真实数据推导用户画像：

```python
# 1. 从交互日志计算行为特征
user_log = log[log["user_id"] == uid]
like_rate    = user_log["is_like"].mean()      # 该用户的真实点赞率
comment_rate = user_log["is_comment"].mean()    # 该用户的真实评论率
skip_rate    = (user_log["play_time_ms"] < 2000).mean()  # 跳过率
avg_wr       = (user_log["play_time_ms"] / user_log["duration_ms"]).mean()

# 2. 从用户特征推导社交特征
active_degree = user_features["user_active_degree"]  # full/high/middle/low
→ 映射为 ActivityLevel.HIGH / MEDIUM / LOW

# 3. 从观看品类分布推导兴趣向量
categories_watched = [tag_map[vid] for vid in user_log["video_id"]]
→ 计算各品类占比 → 兴趣向量 interest_vector

# 4. 从跳过率推导滑动风格
if skip_rate > 0.6:  swipe_profile = FAST_SCANNER
elif skip_rate < 0.3: swipe_profile = BINGE_VIEWER
else:                  swipe_profile = DELIBERATE_WATCHER
```

**关键**: 每个 Agent 的 `like_rate` 等基础参数来自**该用户的真实历史**，而非全局平均或随机生成。这样校准层就能用 persona 自身的 base_rate 做精准门控。

### 4.3 内容池构建细节

```python
# 从真实视频构建
for video in video_features_basic:
    Video(
        video_id   = video["video_id"],
        category   = map_kuaishou_tag(video["tag"]),  # "搞笑" → "comedy"
        creator_id = video["author_id"],
        duration   = video["video_duration"] / 1000,   # ms → s
        like_count = video_stats["like_cnt"],           # 日均点赞
        quality    = video_stats["play_progress"],      # 平均播放进度
        embedding  = 基于品类的 32 维向量,
    )
```

品类标签映射（快手中文 → 系统英文分类）：

| 快手标签 | 映射 | 快手标签 | 映射 |
|---------|------|---------|------|
| 搞笑 | comedy | 美食 | food |
| 舞蹈 | dance | 音乐 | music |
| 游戏 | gaming | 宠物/萌宠 | pets |
| 穿搭 | fashion | 美妆/颜值 | beauty |
| 体育 | sports | 知识 | education |
| 科技 | tech | 旅行 | travel |
| 健身 | fitness | 手工 | diy |
| 新闻 | news | 日常/生活 | vlog |
| 影视/情感 | drama | 动漫/二次元 | animation |
| 汽车 | cars | 自然/摄影 | nature |

### 4.4 Ground Truth 提取

从随机曝光日志（`is_rand=1`）计算无偏的真实分布：

```python
log_random = log[log["is_rand"] == 1]

ground_truth = {
    "like_rate":       0.0048,   # 0.48% 的视频被点赞
    "comment_rate":    0.0003,   # 0.03%
    "forward_rate":    0.0003,   # 0.03%
    "follow_rate":     0.0003,   # 0.03%
    "skip_rate(<2s)":  0.4798,   # 48% 的视频在 2 秒内被跳过
    "completion(>80%)": 0.0778,  # 7.8% 的视频被看完 80% 以上
    "avg_watch_ratio": 0.445,    # 平均观看比例 44.5%
}
```

---

## 五、验证方法

### 5.1 分布级对比

不是比较"某个用户某次看了多久"，而是比较**群体行为分布**。

```
真实数据的 watch_ratio 分布:
  ████████████████████  48% 在 0-0.1（秒跳）
  ████                   8% 在 0.1-0.3
  ██                     5% 在 0.3-0.5
  ██                     5% 在 0.5-0.8
  ███████                8% 在 0.8-1.0（完播）
  ████████              26% 在 >1.0（重播）

模拟数据的 watch_ratio 分布:
  ???  → 用 Wasserstein 距离量化差异
```

### 5.2 三个量化指标

| 指标 | 含义 | 值域 | 越小越好 |
|------|------|------|---------|
| **Wasserstein 距离** | 两个分布之间的"搬土距离" | [0, ∞) | 是 |
| **JS 散度** | 两个分布的信息论距离 | [0, ln2] | 是 |
| **KS 检验 p 值** | 两个分布来自同一总体的概率 | [0, 1] | 越大越好 |

当前基线: Wasserstein=0.218, JS=0.097

### 5.3 逐步改进追踪

每次优化后重新跑验证，追踪这三个数字：

| 版本 | 改动 | Wasserstein | JS 散度 | 备注 |
|------|------|-------------|---------|------|
| v1（当前） | 首次接入 KuaiRand | 0.218 | 0.097 | 基线 |
| v2 | 调整校准强度 | ? | ? | 待做 |
| v3 | 按品类校准 | ? | ? | 待做 |

目标: Wasserstein < 0.1, JS < 0.05

---

## 六、同族数据集 KuaiRec

KuaiRec 是快手发布的另一个数据集，与 KuaiRand 同源但用途不同。

| 维度 | KuaiRand | KuaiRec |
|------|----------|---------|
| 核心特点 | 随机曝光，无偏评估 | **99.6% 密度**，近乎完全观测 |
| 反馈信号 | 12 种（click/like/comment...） | 1 种（watch_ratio 连续值） |
| 社交图谱 | 无（仅统计量） | **有**（friend_list） |
| 密度 | 稀疏（正常推荐日志） | 1,411 用户 × 3,327 视频 = 99.6% 覆盖 |
| 规模 | 27K 用户, 3.22 亿交互 | 7K 用户, 1,250 万交互 |
| 适用 | 去偏、RL、离线策略评估 | 离线 AB 测试、无缺失数据评估 |

**99.6% 密度的含义**: 1,411 个用户几乎看了全部 3,327 个视频。缺失的 0.4% 是因为用户屏蔽了某些创作者导致无法展示。这意味着可以做**无缺失数据偏差**的离线评估。

**我们的使用策略**:
- **KuaiRand**: 用于模拟验证（有随机曝光 ground truth + 丰富的 12 种反馈信号）
- **KuaiRec**: 备用，适合做密集矩阵上的离线 AB（当需要评估"如果这个用户看了这个视频会怎样"时）

---

## 七、数据文件位置与加载

### 文件位置

```
sim3/data/KuaiRand-Pure/data/
  ├── log_random_4_22_to_5_08_pure.csv       # 随机曝光日志（118万条）
  ├── log_standard_4_08_to_4_21_pure.csv     # 正常推荐日志（期间1）
  ├── log_standard_4_22_to_5_08_pure.csv     # 正常推荐日志（期间2）
  ├── user_features_pure.csv                  # 用户特征（27,285 用户 × 31 字段）
  ├── video_features_basic_pure.csv           # 视频基础特征（7,583 视频 × 12 字段）
  └── video_features_statistic_pure.csv       # 视频统计特征（7,583 视频 × 50+ 字段）
```

### 代码加载

```python
from src.data.kuairand_loader import KuaiRandLoader

loader = KuaiRandLoader("data/KuaiRand-Pure/data")

# 获取 ground truth 统计
gt = loader.get_ground_truth_stats()
print(f"真实 like_rate: {gt.like_rate:.4f}")  # 0.0048

# 构建 Persona（从真实用户数据）
personas = loader.build_personas(n_users=100)

# 构建内容池（从真实视频数据）
content_pool = loader.build_content_pool_from_data()

# 获取真实 watch_ratio 分布（用于对比）
real_watch_ratios = loader.get_real_watch_ratios()
```

### 验证运行

```bash
DEEPSEEK_API_KEY="your-key" python3 scripts/validate_with_kuairand.py
# 输出到 output/kuairand_validation/validation_report.json
```
