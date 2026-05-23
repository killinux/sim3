from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from src.agents.persona import CATEGORIES


@dataclass
class Video:
    video_id: str
    category: str
    creator_id: str
    duration_seconds: float
    tags: list[str] = field(default_factory=list)
    like_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    quality_score: float = 0.5
    embedding: np.ndarray | None = None

    def to_prompt_text(self) -> str:
        stats = f"{self.like_count:,} likes, {self.comment_count:,} comments"
        return (
            f"[{self.category}] by creator_{self.creator_id} "
            f"({self.duration_seconds:.0f}s) | Tags: {', '.join(self.tags)} | {stats}"
        )


class ContentPool:
    def __init__(self, num_videos: int = 5000, num_categories: int = 20, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.videos: dict[str, Video] = {}
        self.category_index: dict[str, list[str]] = {}
        self._generate_synthetic_pool(num_videos, num_categories)

    def _generate_synthetic_pool(self, num_videos: int, num_categories: int):
        categories = CATEGORIES[:num_categories]
        category_weights = self.np_rng.dirichlet(np.ones(num_categories) * 2)

        tags_per_category = {
            "comedy": ["funny", "humor", "sketch", "prank", "standup"],
            "food": ["recipe", "cooking", "mukbang", "restaurant", "street_food"],
            "dance": ["choreography", "challenge", "tiktok_dance", "freestyle"],
            "music": ["cover", "original", "instrument", "singing", "remix"],
            "gaming": ["gameplay", "walkthrough", "esports", "mobile_game", "review"],
            "pets": ["cute", "dog", "cat", "training", "rescue"],
            "fashion": ["ootd", "haul", "styling", "trend", "luxury"],
            "beauty": ["makeup", "skincare", "tutorial", "review", "routine"],
            "sports": ["highlights", "training", "extreme", "basketball", "soccer"],
            "education": ["learning", "science", "history", "language", "tips"],
            "tech": ["gadget", "review", "unboxing", "coding", "ai"],
            "travel": ["vlog", "guide", "adventure", "budget", "luxury_travel"],
            "fitness": ["workout", "yoga", "diet", "transformation", "gym"],
            "diy": ["craft", "home", "repair", "upcycle", "tutorial"],
            "news": ["breaking", "analysis", "opinion", "local", "world"],
            "vlog": ["daily", "lifestyle", "grwm", "routine", "story"],
            "drama": ["series", "short_film", "emotional", "romance", "thriller"],
            "animation": ["cartoon", "anime", "3d", "stop_motion", "meme"],
            "cars": ["review", "racing", "modification", "electric", "luxury_car"],
            "nature": ["wildlife", "landscape", "ocean", "documentary", "asmr"],
        }

        embedding_dim = 32
        category_centroids = {}
        for i, cat in enumerate(categories):
            centroid = np.zeros(embedding_dim)
            centroid[i % embedding_dim] = 1.0
            centroid += self.np_rng.randn(embedding_dim) * 0.3
            category_centroids[cat] = centroid / np.linalg.norm(centroid)

        num_creators = num_videos // 10
        for i in range(num_videos):
            cat_idx = self.np_rng.choice(len(categories), p=category_weights)
            cat = categories[cat_idx]
            vid_id = f"v_{i:06d}"
            creator_id = str(self.rng.randint(0, num_creators - 1))

            duration = self.rng.choice([
                self.rng.uniform(5, 15),
                self.rng.uniform(15, 60),
                self.rng.uniform(60, 180),
            ])

            available_tags = tags_per_category.get(cat, ["general"])
            num_tags = self.rng.randint(1, min(3, len(available_tags)))
            tags = self.rng.sample(available_tags, num_tags)

            quality = max(0.0, min(1.0, self.rng.gauss(0.5, 0.2)))
            popularity_base = quality * self.rng.lognormvariate(5, 2)
            likes = max(0, int(popularity_base))
            comments = max(0, int(likes * self.rng.uniform(0.01, 0.1)))
            shares = max(0, int(likes * self.rng.uniform(0.005, 0.05)))

            emb = category_centroids[cat] + self.np_rng.randn(embedding_dim) * 0.15
            emb = emb / np.linalg.norm(emb)

            video = Video(
                video_id=vid_id,
                category=cat,
                creator_id=creator_id,
                duration_seconds=round(duration, 1),
                tags=tags,
                like_count=likes,
                comment_count=comments,
                share_count=shares,
                quality_score=quality,
                embedding=emb,
            )
            self.videos[vid_id] = video
            self.category_index.setdefault(cat, []).append(vid_id)

    def get_video(self, video_id: str) -> Video | None:
        return self.videos.get(video_id)

    def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        ids = list(self.videos.keys())
        embeddings = np.array([self.videos[vid_id].embedding for vid_id in ids])
        return ids, embeddings

    def sample_videos(self, n: int, category: str | None = None) -> list[Video]:
        if category and category in self.category_index:
            pool = self.category_index[category]
        else:
            pool = list(self.videos.keys())
        sampled_ids = self.rng.sample(pool, min(n, len(pool)))
        return [self.videos[vid_id] for vid_id in sampled_ids]
