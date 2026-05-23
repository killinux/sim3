from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Interaction:
    video_id: str
    category: str
    duration_seconds: float
    watch_ratio: float
    liked: bool = False
    commented: bool = False
    shared: bool = False
    followed_creator: bool = False
    timestamp: float = 0.0

    def to_text(self) -> str:
        parts = [f"Video({self.category}, {self.duration_seconds:.0f}s)"]
        if self.watch_ratio < 0.15:
            parts.append("skipped")
        elif self.watch_ratio < 0.8:
            parts.append(f"watched {self.watch_ratio:.0%}")
        else:
            parts.append("watched fully")
        actions = []
        if self.liked:
            actions.append("liked")
        if self.commented:
            actions.append("commented")
        if self.shared:
            actions.append("shared")
        if self.followed_creator:
            actions.append("followed creator")
        if actions:
            parts.append(", ".join(actions))
        return " | ".join(parts)

    @property
    def engagement_score(self) -> float:
        score = min(self.watch_ratio, 1.0) * 0.4
        if self.liked:
            score += 0.2
        if self.commented:
            score += 0.2
        if self.shared:
            score += 0.15
        if self.followed_creator:
            score += 0.05
        return score


@dataclass
class EpisodicEvent:
    description: str
    interaction: Interaction
    importance: float = 1.0


class AgentMemory:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.working_memory: list[Interaction] = []
        self.episodic_memory: list[EpisodicEvent] = []
        self.session_summaries: list[str] = []
        self._interaction_count = 0

    def add_interaction(self, interaction: Interaction):
        self.working_memory.append(interaction)
        if len(self.working_memory) > self.window_size:
            self.working_memory.pop(0)
        self._interaction_count += 1

        if interaction.engagement_score > 0.5:
            self.episodic_memory.append(
                EpisodicEvent(
                    description=interaction.to_text(),
                    interaction=interaction,
                    importance=interaction.engagement_score,
                )
            )

    def get_working_memory_text(self, max_items: int | None = None) -> str:
        items = self.working_memory
        if max_items:
            items = items[-max_items:]
        if not items:
            return "No recent viewing history."
        lines = [f"  {i + 1}. {item.to_text()}" for i, item in enumerate(items)]
        return "Recent viewing history:\n" + "\n".join(lines)

    def get_session_context(self) -> str:
        parts = []
        if self.session_summaries:
            parts.append(f"Previous sessions: {len(self.session_summaries)}")
            if self.session_summaries:
                parts.append(f"Last session: {self.session_summaries[-1]}")
        parts.append(f"Videos watched this session: {self._interaction_count}")
        return "\n".join(parts) if parts else ""

    def add_session_summary(self, summary: str):
        self.session_summaries.append(summary)
        if len(self.session_summaries) > 10:
            self.session_summaries.pop(0)

    def reset_session(self):
        self._interaction_count = 0

    @property
    def total_interactions(self) -> int:
        return len(self.working_memory) + sum(
            1 for _ in self.episodic_memory
            if _ not in self.working_memory
        )

    def get_engagement_stats(self) -> dict[str, float]:
        if not self.working_memory:
            return {"avg_watch_ratio": 0.0, "like_rate": 0.0, "skip_rate": 0.0}
        n = len(self.working_memory)
        return {
            "avg_watch_ratio": sum(i.watch_ratio for i in self.working_memory) / n,
            "like_rate": sum(1 for i in self.working_memory if i.liked) / n,
            "skip_rate": sum(1 for i in self.working_memory if i.watch_ratio < 0.15) / n,
        }
