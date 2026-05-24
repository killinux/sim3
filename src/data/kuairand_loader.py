from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.agents.persona import (
    CATEGORIES,
    ActivityLevel,
    BehavioralSignature,
    ConformityLevel,
    Demographics,
    DiversityLevel,
    Preferences,
    SwipeProfile,
    UserPersona,
)
from src.recommendation.content_pool import ContentPool, Video


@dataclass
class KuaiRandStats:
    """Ground truth statistics from KuaiRand for validation."""
    n_interactions: int
    like_rate: float
    comment_rate: float
    forward_rate: float
    follow_rate: float
    hate_rate: float
    click_rate: float
    skip_rate_2s: float
    completion_rate_80: float
    avg_watch_ratio: float
    median_watch_ratio: float
    watch_ratio_distribution: dict


class KuaiRandLoader:
    def __init__(self, data_dir: str = "data/KuaiRand-Pure/data"):
        self.data_dir = data_dir
        self._log_random = None
        self._log_standard = None
        self._user_features = None
        self._video_basic = None
        self._video_stats = None

    def _load_random_log(self) -> pd.DataFrame:
        if self._log_random is None:
            path = os.path.join(self.data_dir, "log_random_4_22_to_5_08_pure.csv")
            self._log_random = pd.read_csv(path)
            self._log_random["watch_ratio"] = (
                self._log_random["play_time_ms"]
                / self._log_random["duration_ms"].clip(lower=1)
            ).clip(0, 10)
        return self._log_random

    def _load_standard_log(self) -> pd.DataFrame:
        if self._log_standard is None:
            path = os.path.join(self.data_dir, "log_standard_4_08_to_4_21_pure.csv")
            self._log_standard = pd.read_csv(path)
            self._log_standard["watch_ratio"] = (
                self._log_standard["play_time_ms"]
                / self._log_standard["duration_ms"].clip(lower=1)
            ).clip(0, 10)
        return self._log_standard

    def _load_user_features(self) -> pd.DataFrame:
        if self._user_features is None:
            path = os.path.join(self.data_dir, "user_features_pure.csv")
            self._user_features = pd.read_csv(path)
        return self._user_features

    def _load_video_basic(self) -> pd.DataFrame:
        if self._video_basic is None:
            path = os.path.join(self.data_dir, "video_features_basic_pure.csv")
            self._video_basic = pd.read_csv(path)
        return self._video_basic

    def _load_video_stats(self) -> pd.DataFrame:
        if self._video_stats is None:
            path = os.path.join(self.data_dir, "video_features_statistic_pure.csv")
            self._video_stats = pd.read_csv(path)
        return self._video_stats

    def get_ground_truth_stats(self) -> KuaiRandStats:
        log = self._load_random_log()
        n = len(log)
        wr = log["watch_ratio"].values
        return KuaiRandStats(
            n_interactions=n,
            like_rate=log["is_like"].mean(),
            comment_rate=log["is_comment"].mean(),
            forward_rate=log["is_forward"].mean(),
            follow_rate=log["is_follow"].mean(),
            hate_rate=log["is_hate"].mean(),
            click_rate=log["is_click"].mean(),
            skip_rate_2s=(log["play_time_ms"] < 2000).mean(),
            completion_rate_80=(wr > 0.8).mean(),
            avg_watch_ratio=float(np.mean(wr)),
            median_watch_ratio=float(np.median(wr)),
            watch_ratio_distribution={
                "p10": float(np.percentile(wr, 10)),
                "p25": float(np.percentile(wr, 25)),
                "p50": float(np.percentile(wr, 50)),
                "p75": float(np.percentile(wr, 75)),
                "p90": float(np.percentile(wr, 90)),
            },
        )

    def get_real_watch_ratios(self) -> np.ndarray:
        log = self._load_random_log()
        return log["watch_ratio"].clip(0, 5).values

    def build_personas(self, n_users: int = 100, seed: int = 42) -> list[UserPersona]:
        import random
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        log = self._load_random_log()
        uf = self._load_user_features()

        user_ids = log["user_id"].unique()
        if n_users < len(user_ids):
            chosen = rng.sample(list(user_ids), n_users)
        else:
            chosen = list(user_ids)

        vb = self._load_video_basic()
        tag_map = {}
        for _, row in vb.iterrows():
            tag_str = str(row.get("tag", ""))
            if tag_str and tag_str != "nan":
                tag_map[row["video_id"]] = tag_str.split(",")[0].strip() if tag_str else "unknown"

        personas = []
        for uid in chosen:
            user_log = log[log["user_id"] == uid]
            user_feat_row = uf[uf["user_id"] == uid]

            n_interactions = len(user_log)
            if n_interactions == 0:
                continue

            like_rate = user_log["is_like"].mean()
            comment_rate = user_log["is_comment"].mean()
            forward_rate = user_log["is_forward"].mean()
            follow_rate = user_log["is_follow"].mean()
            avg_wr = user_log["watch_ratio"].mean()
            skip_rate = (user_log["play_time_ms"] < 2000).mean()

            active_degree = "middle_active"
            if not user_feat_row.empty:
                active_degree = user_feat_row.iloc[0].get("user_active_degree", "middle_active")

            if active_degree in ("full_active", "high_active"):
                activity = ActivityLevel.HIGH
            elif active_degree == "middle_active":
                activity = ActivityLevel.MEDIUM
            else:
                activity = ActivityLevel.LOW

            video_ids = user_log["video_id"].values
            categories_seen = [tag_map.get(vid, "unknown") for vid in video_ids]
            cat_counts = {}
            for c in categories_seen:
                cat_counts[c] = cat_counts.get(c, 0) + 1
            n_distinct_cats = len(cat_counts)
            if n_distinct_cats >= 8:
                diversity = DiversityLevel.BROAD
            elif n_distinct_cats >= 4:
                diversity = DiversityLevel.MODERATE
            else:
                diversity = DiversityLevel.NARROW

            interest = {}
            total_cat = sum(cat_counts.values())
            for cat_name, cnt in cat_counts.items():
                mapped = _map_kuaishou_tag(cat_name)
                interest[mapped] = interest.get(mapped, 0) + cnt / total_cat
            for cat in CATEGORIES:
                if cat not in interest:
                    interest[cat] = 0.01

            conformity = rng.choice(list(ConformityLevel))

            sessions_per_day = max(1.0, n_interactions / 14 / 5)
            videos_per_session = min(50, n_interactions / max(1, sessions_per_day * 14))

            peak = [0.02] * 24
            for _, row in user_log.iterrows():
                hm = int(row.get("hourmin", 1200))
                hour = hm // 100
                if 0 <= hour < 24:
                    peak[hour] += 1
            total_p = sum(peak)
            peak = [p / total_p for p in peak]

            swipe = SwipeProfile.DELIBERATE_WATCHER
            if skip_rate > 0.6:
                swipe = SwipeProfile.FAST_SCANNER
            elif skip_rate < 0.3:
                swipe = SwipeProfile.BINGE_VIEWER

            persona = UserPersona(
                demographics=Demographics(
                    user_id=f"ku_{uid}",
                    age_bucket="unknown",
                    gender="unknown",
                    city_tier=3,
                    device_type="unknown",
                ),
                preferences=Preferences(
                    interest_vector=interest,
                    novelty_seeking=rng.uniform(0.2, 0.8),
                    trend_sensitivity=rng.uniform(0.2, 0.8),
                ),
                behavior=BehavioralSignature(
                    avg_session_minutes=rng.uniform(3, 15),
                    sessions_per_day=sessions_per_day,
                    videos_per_session=videos_per_session,
                    like_rate=max(0.001, like_rate),
                    comment_rate=max(0.0001, comment_rate),
                    share_rate=max(0.0001, forward_rate),
                    follow_rate=max(0.0001, follow_rate),
                    peak_hours=peak,
                    swipe_profile=swipe,
                ),
                activity=activity,
                conformity=conformity,
                diversity=diversity,
            )
            personas.append(persona)

        return personas

    def build_content_pool_from_data(self) -> ContentPool:
        vb = self._load_video_basic()
        vs = self._load_video_stats()

        pool = ContentPool.__new__(ContentPool)
        pool.videos = {}
        pool.category_index = {}
        pool.rng = __import__("random").Random(42)
        pool.np_rng = np.random.RandomState(42)

        merged = vb.merge(vs, on="video_id", how="left")

        embedding_dim = 32
        tag_centroids = {}

        for _, row in merged.iterrows():
            vid_id = f"kv_{int(row['video_id'])}"
            tag_str = str(row.get("tag", ""))
            tags = [t.strip() for t in tag_str.split(",") if t.strip()] if tag_str and tag_str != "nan" else ["unknown"]
            category = _map_kuaishou_tag(tags[0] if tags else "unknown")
            duration_ms = row.get("video_duration", 10000)
            duration_s = max(1.0, duration_ms / 1000.0)
            author_id = str(int(row.get("author_id", 0)))

            like_cnt = int(row.get("like_cnt", 0)) if not pd.isna(row.get("like_cnt")) else 0
            comment_cnt = int(row.get("comment_cnt", 0)) if not pd.isna(row.get("comment_cnt")) else 0
            share_cnt = int(row.get("share_cnt", 0)) if not pd.isna(row.get("share_cnt")) else 0
            play_progress = row.get("play_progress", 0.3)
            quality = float(play_progress) if not pd.isna(play_progress) else 0.3

            if category not in tag_centroids:
                centroid = np.zeros(embedding_dim)
                idx = CATEGORIES.index(category) % embedding_dim if category in CATEGORIES else hash(category) % embedding_dim
                centroid[idx] = 1.0
                centroid += pool.np_rng.randn(embedding_dim) * 0.2
                tag_centroids[category] = centroid / np.linalg.norm(centroid)

            emb = tag_centroids[category] + pool.np_rng.randn(embedding_dim) * 0.1
            emb = emb / np.linalg.norm(emb)

            video = Video(
                video_id=vid_id,
                category=category,
                creator_id=author_id,
                duration_seconds=round(duration_s, 1),
                tags=tags[:3],
                like_count=like_cnt,
                comment_count=comment_cnt,
                share_count=share_cnt,
                quality_score=min(1.0, quality),
                embedding=emb,
            )
            pool.videos[vid_id] = video
            pool.category_index.setdefault(category, []).append(vid_id)

        return pool


_KUAISHOU_TAG_MAP = {
    "搞笑": "comedy", "美食": "food", "舞蹈": "dance", "音乐": "music",
    "游戏": "gaming", "宠物": "pets", "穿搭": "fashion", "美妆": "beauty",
    "体育": "sports", "知识": "education", "科技": "tech", "旅行": "travel",
    "健身": "fitness", "手工": "diy", "新闻": "news", "日常": "vlog",
    "影视": "drama", "动漫": "animation", "汽车": "cars", "自然": "nature",
    "生活": "vlog", "颜值": "beauty", "情感": "drama", "萌宠": "pets",
    "二次元": "animation", "才艺": "dance", "摄影": "nature",
}


def _map_kuaishou_tag(tag: str) -> str:
    tag = tag.strip().lower()
    if tag in _KUAISHOU_TAG_MAP:
        return _KUAISHOU_TAG_MAP[tag]
    for key, val in _KUAISHOU_TAG_MAP.items():
        if key in tag:
            return val
    mapped_cats = list(set(_KUAISHOU_TAG_MAP.values()))
    return mapped_cats[hash(tag) % len(mapped_cats)]
