from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

from src.agents.memory import Interaction
from src.agents.persona import UserPersona, generate_random_persona
from src.agents.user_agent import UserAgent
from src.recommendation.content_pool import ContentPool
from src.recommendation.surrogate_recsys import SurrogateRecSys
from src.simulation.scheduler import LLMScheduler

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    num_agents: int = 100
    num_simulated_days: int = 1
    max_videos_per_session: int = 50
    max_sessions_per_day: int = 5
    memory_window: int = 30
    seed: int = 42
    content_pool_size: int = 5000
    num_categories: int = 20
    rec_top_k: int = 10
    rec_epsilon: float = 0.1
    rec_diversity_weight: float = 0.2
    ab_test_enabled: bool = False
    ab_treatment_ratio: float = 0.5
    ab_treatment_config: dict = field(default_factory=dict)


@dataclass
class InteractionRecord:
    user_id: str
    video_id: str
    category: str
    duration_seconds: float
    watch_ratio: float
    liked: bool
    commented: bool
    shared: bool
    followed: bool
    session_id: int
    position_in_session: int
    fatigue_at_decision: float
    variant: str
    simulated_day: int
    timestamp: float


class SimulationEngine:
    def __init__(self, config: SimulationConfig, scheduler: LLMScheduler):
        self.config = config
        self.scheduler = scheduler
        self.rng = random.Random(config.seed)

        self.content_pool = ContentPool(
            num_videos=config.content_pool_size,
            num_categories=config.num_categories,
            seed=config.seed,
        )
        self.recsys = SurrogateRecSys(
            content_pool=self.content_pool,
            top_k=config.rec_top_k,
            epsilon=config.rec_epsilon,
            diversity_weight=config.rec_diversity_weight,
            seed=config.seed,
        )

        self.agents: dict[str, UserAgent] = {}
        self.agent_variants: dict[str, str] = {}
        self.interaction_log: list[InteractionRecord] = []
        self._session_counter = 0

    def initialize_agents(self, personas: list[UserPersona] | None = None):
        if personas:
            for persona in personas:
                agent = UserAgent(persona, memory_window=self.config.memory_window)
                self.agents[agent.user_id] = agent
        else:
            for i in range(self.config.num_agents):
                persona = generate_random_persona(f"u_{i:04d}", self.rng)
                agent = UserAgent(persona, memory_window=self.config.memory_window)
                self.agents[agent.user_id] = agent

        if self.config.ab_test_enabled:
            agent_ids = list(self.agents.keys())
            self.rng.shuffle(agent_ids)
            split = int(len(agent_ids) * self.config.ab_treatment_ratio)
            for uid in agent_ids[:split]:
                self.agent_variants[uid] = "treatment"
            for uid in agent_ids[split:]:
                self.agent_variants[uid] = "control"
        else:
            for uid in self.agents:
                self.agent_variants[uid] = "control"

        logger.info(
            "Initialized %d agents (treatment=%d, control=%d)",
            len(self.agents),
            sum(1 for v in self.agent_variants.values() if v == "treatment"),
            sum(1 for v in self.agent_variants.values() if v == "control"),
        )

    async def run(self) -> list[InteractionRecord]:
        UserAgent.reset_interest_tracker()
        for day in range(self.config.num_simulated_days):
            logger.info("=== Simulated Day %d ===", day + 1)
            await self._simulate_day(day)
            logger.info(
                "Day %d complete: %d total interactions",
                day + 1,
                len(self.interaction_log),
            )
        return self.interaction_log

    async def _simulate_day(self, day: int):
        agent_list = list(self.agents.values())
        tasks = [self._simulate_agent_day(agent, day) for agent in agent_list]
        await asyncio.gather(*tasks)

    async def _simulate_agent_day(self, agent: UserAgent, day: int):
        num_sessions = max(
            1,
            int(agent.persona.behavior.sessions_per_day + self.rng.uniform(-0.5, 0.5)),
        )
        num_sessions = min(num_sessions, self.config.max_sessions_per_day)

        for _ in range(num_sessions):
            await self._simulate_session(agent, day)

    async def _simulate_session(self, agent: UserAgent, day: int):
        self._session_counter += 1
        session_id = self._session_counter
        agent.start_session()

        variant = self.agent_variants[agent.user_id]
        variant_config = self.config.ab_treatment_config if variant == "treatment" else None

        for position in range(self.config.max_videos_per_session):
            feed = self.recsys.recommend(
                agent.persona,
                watched_ids=agent.watched_video_ids,
                variant_config=variant_config,
            )
            if not feed:
                break

            video = feed[0]
            system_prompt, user_prompt = agent.build_decision_prompt(video)

            llm_response = await self.scheduler.request(system_prompt, user_prompt)
            decision = agent.parse_llm_response(llm_response, video)
            fatigue_before = agent.fatigue
            interaction = agent.process_decision(decision, video)

            record = InteractionRecord(
                user_id=agent.user_id,
                video_id=video.video_id,
                category=video.category,
                duration_seconds=video.duration_seconds,
                watch_ratio=interaction.watch_ratio,
                liked=interaction.liked,
                commented=interaction.commented,
                shared=interaction.shared,
                followed=interaction.followed_creator,
                session_id=session_id,
                position_in_session=position,
                fatigue_at_decision=fatigue_before,
                variant=variant,
                simulated_day=day,
                timestamp=time.time(),
            )
            self.interaction_log.append(record)

            if agent.should_end_session(decision):
                break

        agent.end_session()

    def get_interaction_summary(self) -> dict:
        if not self.interaction_log:
            return {"total_interactions": 0}

        records = self.interaction_log
        n = len(records)
        return {
            "total_interactions": n,
            "unique_users": len({r.user_id for r in records}),
            "unique_videos": len({r.video_id for r in records}),
            "avg_watch_ratio": sum(r.watch_ratio for r in records) / n,
            "like_rate": sum(1 for r in records if r.liked) / n,
            "comment_rate": sum(1 for r in records if r.commented) / n,
            "share_rate": sum(1 for r in records if r.shared) / n,
            "follow_rate": sum(1 for r in records if r.followed) / n,
            "avg_videos_per_session": n / len({r.session_id for r in records}),
            "llm_stats": self.scheduler.get_stats(),
        }
