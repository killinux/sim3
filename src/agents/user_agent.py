from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from src.agents.memory import AgentMemory, Interaction
from src.agents.persona import UserPersona
from src.recommendation.content_pool import Video


class UserAction(str, Enum):
    SKIP = "skip"
    WATCH_PARTIAL = "watch_partial"
    WATCH_FULL = "watch_full"
    LIKE = "like"
    COMMENT = "comment"
    SHARE = "share"
    FOLLOW = "follow"
    END_SESSION = "end_session"


@dataclass
class AgentDecision:
    action: UserAction
    watch_ratio: float = 0.0
    liked: bool = False
    commented: bool = False
    shared: bool = False
    followed: bool = False
    continue_browsing: bool = True
    reasoning: str = ""


DECISION_PROMPT = """\
You are scrolling through a short video feed. Here is the next video:

{video_info}

{memory_context}

Current session state:
- Videos watched this session: {videos_watched}
- Current fatigue level: {fatigue:.2f} (0=fresh, 1=exhausted)

IMPORTANT - Real user engagement benchmarks (be realistic, not generous):
- Most users SKIP 40-60% of videos after watching <2 seconds
- Only 5-8% of watched videos get a LIKE
- Only 0.5-2% get a COMMENT
- Only 0.3-1% get a SHARE
- Only 0.1-0.5% lead to FOLLOW
- Users are selective and picky, not enthusiastic about everything

Based on your personality and current state, respond with a JSON object:
{{
  "decision": "skip" | "watch_partial" | "watch_full",
  "watch_percent": <integer 0-100, how much of the video you'd watch>,
  "like": true | false,
  "comment": true | false,
  "share": true | false,
  "follow_creator": true | false,
  "continue": true | false,
  "reason": "<one sentence explaining your decision>"
}}

Respond with ONLY the JSON object, no other text."""


class UserAgent:
    def __init__(self, persona: UserPersona, memory_window: int = 30):
        self.persona = persona
        self.memory = AgentMemory(window_size=memory_window)
        self.session_video_count = 0
        self.fatigue = 0.0
        self.fatigue_rate = 0.02
        self.watched_video_ids: set[str] = set()

    @property
    def user_id(self) -> str:
        return self.persona.demographics.user_id

    def build_decision_prompt(self, video: Video) -> tuple[str, str]:
        system = self.persona.build_system_prompt()
        memory_text = self.memory.get_working_memory_text(max_items=10)
        user = DECISION_PROMPT.format(
            video_info=video.to_prompt_text(),
            memory_context=memory_text,
            videos_watched=self.session_video_count,
            fatigue=self.fatigue,
        )
        return system, user

    def parse_llm_response(self, response: str, video: Video) -> AgentDecision:
        try:
            cleaned = response.strip()
            json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group()
            data = json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            return self._fallback_decision(video)

        decision_str = data.get("decision", "skip")
        watch_pct = data.get("watch_percent", 0)

        if decision_str == "skip":
            action = UserAction.SKIP
            watch_ratio = min(watch_pct, 10) / 100.0
        elif decision_str == "watch_partial":
            action = UserAction.WATCH_PARTIAL
            watch_ratio = max(0.1, min(watch_pct, 95)) / 100.0
        elif decision_str == "watch_full":
            action = UserAction.WATCH_FULL
            watch_ratio = max(0.8, min(watch_pct, 150)) / 100.0
        else:
            action = UserAction.SKIP
            watch_ratio = 0.05

        liked = bool(data.get("like", False))
        commented = bool(data.get("comment", False))
        shared = bool(data.get("share", False))
        followed = bool(data.get("follow_creator", False))

        liked, commented, shared, followed = self._calibrate_engagement(
            liked, commented, shared, followed, watch_ratio, video,
        )

        return AgentDecision(
            action=action,
            watch_ratio=watch_ratio,
            liked=liked,
            commented=commented,
            shared=shared,
            followed=followed,
            continue_browsing=bool(data.get("continue", True)),
            reasoning=data.get("reason", ""),
        )

    def _calibrate_engagement(
        self,
        liked: bool,
        commented: bool,
        shared: bool,
        followed: bool,
        watch_ratio: float,
        video: Video,
    ) -> tuple[bool, bool, bool, bool]:
        import random
        rng = random.Random(hash((self.user_id, video.video_id, self.session_video_count)))

        base_like = self.persona.behavior.like_rate
        base_comment = self.persona.behavior.comment_rate
        base_share = self.persona.behavior.share_rate
        base_follow = self.persona.behavior.follow_rate

        engagement_mult = max(0.2, min(3.0, watch_ratio * 2))

        if liked:
            keep_prob = base_like * engagement_mult / max(base_like * engagement_mult, 0.15)
            liked = rng.random() < keep_prob
        if commented:
            keep_prob = base_comment * engagement_mult / max(base_comment * engagement_mult, 0.05)
            commented = rng.random() < keep_prob
        if shared:
            keep_prob = base_share * engagement_mult / max(base_share * engagement_mult, 0.03)
            shared = rng.random() < keep_prob
        if followed:
            keep_prob = base_follow * engagement_mult / max(base_follow * engagement_mult, 0.02)
            followed = rng.random() < keep_prob

        if watch_ratio < 0.3:
            liked = False
            commented = False
            shared = False
            followed = False

        return liked, commented, shared, followed

    def _fallback_decision(self, video: Video) -> AgentDecision:
        import random
        rng = random.Random(hash(self.user_id + video.video_id))
        interest = self.persona.preferences.interest_vector.get(video.category, 0.05)

        if interest < 0.03 or self.fatigue > 0.8:
            return AgentDecision(action=UserAction.SKIP, watch_ratio=0.05)

        watch_ratio = min(1.0, interest * 2 + rng.uniform(-0.2, 0.2))
        return AgentDecision(
            action=UserAction.WATCH_FULL if watch_ratio > 0.7 else UserAction.WATCH_PARTIAL,
            watch_ratio=watch_ratio,
            liked=rng.random() < self.persona.behavior.like_rate * (1 + watch_ratio),
            commented=rng.random() < self.persona.behavior.comment_rate,
            shared=rng.random() < self.persona.behavior.share_rate,
            followed=rng.random() < self.persona.behavior.follow_rate,
            continue_browsing=self.fatigue < 0.7,
        )

    def process_decision(self, decision: AgentDecision, video: Video) -> Interaction:
        interaction = Interaction(
            video_id=video.video_id,
            category=video.category,
            duration_seconds=video.duration_seconds,
            watch_ratio=decision.watch_ratio,
            liked=decision.liked,
            commented=decision.commented,
            shared=decision.shared,
            followed_creator=decision.followed,
            timestamp=0.0,
        )
        self.memory.add_interaction(interaction)
        self.watched_video_ids.add(video.video_id)
        self.session_video_count += 1

        satisfaction = interaction.engagement_score
        self.fatigue += (1.0 - satisfaction) * self.fatigue_rate
        self.fatigue = min(1.0, self.fatigue)

        self.persona.preferences.update_after_watch(video.category, satisfaction)

        return interaction

    def should_end_session(self, decision: AgentDecision) -> bool:
        if not decision.continue_browsing:
            return True
        end_prob = 0.01 * self.fatigue**2 + 0.15 * self.fatigue
        import random
        return random.Random(self.session_video_count).random() < end_prob

    def start_session(self):
        self.session_video_count = 0
        self.fatigue = 0.0
        self.memory.reset_session()

    def end_session(self) -> dict:
        stats = self.memory.get_engagement_stats()
        summary = (
            f"Watched {self.session_video_count} videos. "
            f"Avg completion: {stats['avg_watch_ratio']:.0%}, "
            f"Like rate: {stats['like_rate']:.1%}"
        )
        self.memory.add_session_summary(summary)
        return {
            "user_id": self.user_id,
            "videos_watched": self.session_video_count,
            **stats,
        }
