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

Based on your personality and current state, decide how you'd react.
Think about: does this video match your interests? How is your energy level?

Respond with a JSON object:
{{
  "decision": "skip" | "watch_partial" | "watch_full",
  "watch_percent": <integer 0-100, what percent of the video you'd watch>,
  "interest_level": <integer 1-10, how interesting is this video to you>,
  "continue": true | false,
  "reason": "<one sentence>"
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

        interest = data.get("interest_level", 5)
        interest = max(1, min(10, int(interest)))

        import random as _rnd
        rng = _rnd.Random(hash((self.user_id, video.video_id, self.session_video_count)))

        fatigue_penalty = self.fatigue * 0.3
        effective_interest = max(1, interest - fatigue_penalty * 10)

        category_affinity = self.persona.preferences.interest_vector.get(video.category, 0.05)
        if category_affinity < 0.02:
            effective_interest = max(1, effective_interest - 3)

        skip_threshold = 5 - int(category_affinity * 5)
        skip_threshold = max(3, min(6, skip_threshold))

        if effective_interest <= skip_threshold:
            watch_ratio = max(0.0, rng.expovariate(4.0))
            watch_ratio = min(watch_ratio, 0.12)
            action = UserAction.SKIP
        elif effective_interest <= 7:
            watch_ratio = rng.betavariate(3.0, 2.0) * 0.6 + 0.2
            action = UserAction.WATCH_PARTIAL
        elif effective_interest <= 9:
            watch_ratio = rng.betavariate(4.0, 1.9) * 0.35 + 0.51
            action = UserAction.WATCH_PARTIAL if watch_ratio < 0.8 else UserAction.WATCH_FULL
        else:
            watch_ratio = rng.betavariate(5.0, 2.0) * 0.25 + 0.68
            action = UserAction.WATCH_FULL if watch_ratio >= 0.8 else UserAction.WATCH_PARTIAL

        liked, commented, shared, followed = self._generate_engagement(
            interest, watch_ratio, video,
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

    def _generate_engagement(
        self,
        interest_level: int,
        watch_ratio: float,
        video: Video,
    ) -> tuple[bool, bool, bool, bool]:
        import random
        rng = random.Random(hash((self.user_id, video.video_id, self.session_video_count)))

        if watch_ratio < 0.15:
            return False, False, False, False

        interest_mult = (interest_level / 5.0) ** 1.5
        watch_mult = min(2.0, watch_ratio)
        boost = interest_mult * watch_mult

        like_prob = self.persona.behavior.like_rate * boost
        comment_prob = self.persona.behavior.comment_rate * boost
        share_prob = self.persona.behavior.share_rate * boost
        follow_prob = self.persona.behavior.follow_rate * boost

        liked = rng.random() < like_prob
        commented = rng.random() < comment_prob
        shared = rng.random() < share_prob
        followed = rng.random() < follow_prob

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
