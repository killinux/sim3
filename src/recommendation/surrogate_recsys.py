from __future__ import annotations

import numpy as np

from src.agents.persona import UserPersona, CATEGORIES
from src.recommendation.content_pool import ContentPool, Video


class SurrogateRecSys:
    def __init__(
        self,
        content_pool: ContentPool,
        top_k: int = 10,
        epsilon: float = 0.1,
        diversity_weight: float = 0.2,
        seed: int = 42,
    ):
        self.content_pool = content_pool
        self.top_k = top_k
        self.epsilon = epsilon
        self.diversity_weight = diversity_weight
        self.rng = np.random.RandomState(seed)
        self._video_ids, self._video_embeddings = content_pool.get_all_embeddings()
        self._video_embeddings_normed = (
            self._video_embeddings
            / np.linalg.norm(self._video_embeddings, axis=1, keepdims=True)
        )

    def _build_user_embedding(self, persona: UserPersona) -> np.ndarray:
        embedding_dim = self._video_embeddings.shape[1]
        user_emb = np.zeros(embedding_dim)
        categories = CATEGORIES[: embedding_dim]
        for i, cat in enumerate(categories):
            weight = persona.preferences.interest_vector.get(cat, 0.0)
            user_emb[i % embedding_dim] += weight
        norm = np.linalg.norm(user_emb)
        if norm > 0:
            user_emb = user_emb / norm
        return user_emb

    def recommend(
        self,
        persona: UserPersona,
        watched_ids: set[str] | None = None,
        variant_config: dict | None = None,
    ) -> list[Video]:
        watched_ids = watched_ids or set()
        diversity_w = self.diversity_weight
        eps = self.epsilon
        k = self.top_k

        if variant_config:
            diversity_w = variant_config.get("diversity_weight", diversity_w)
            eps = variant_config.get("epsilon", eps)
            k = variant_config.get("top_k", k)

        user_emb = self._build_user_embedding(persona)

        scores = self._video_embeddings_normed @ user_emb

        for i, vid_id in enumerate(self._video_ids):
            if vid_id in watched_ids:
                scores[i] = -np.inf
            video = self.content_pool.videos[vid_id]
            popularity = np.log1p(video.like_count) / 15.0
            scores[i] = scores[i] * (1 - diversity_w) + popularity * diversity_w

        if self.rng.random() < eps:
            valid_mask = scores > -np.inf
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) < k:
                chosen_indices = valid_indices
            else:
                chosen_indices = self.rng.choice(valid_indices, size=k, replace=False)
        else:
            candidate_k = min(k * 3, len(scores))
            top_indices = np.argpartition(scores, -candidate_k)[-candidate_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            chosen_indices = []
            seen_categories = set()
            for idx in top_indices:
                if len(chosen_indices) >= k:
                    break
                vid = self.content_pool.videos[self._video_ids[idx]]
                cat_count = sum(1 for c in seen_categories if c == vid.category)
                if cat_count < max(2, k // 4):
                    chosen_indices.append(idx)
                    seen_categories.add(vid.category)

            while len(chosen_indices) < k and len(chosen_indices) < len(top_indices):
                for idx in top_indices:
                    if idx not in chosen_indices:
                        chosen_indices.append(idx)
                        if len(chosen_indices) >= k:
                            break

        result = []
        for idx in chosen_indices:
            vid_id = self._video_ids[idx]
            video = self.content_pool.get_video(vid_id)
            if video:
                result.append(video)
        return result
