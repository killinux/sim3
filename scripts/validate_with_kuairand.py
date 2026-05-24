#!/usr/bin/env python3
"""Validate simulation against KuaiRand ground truth."""

import asyncio
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.kuairand_loader import KuaiRandLoader
from src.simulation.engine import SimulationConfig, SimulationEngine
from src.simulation.scheduler import LLMScheduler
from src.validation.distribution_metrics import compare_distributions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    output_dir = "output/kuairand_validation"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading KuaiRand data...")
    loader = KuaiRandLoader("data/KuaiRand-Pure/data")

    gt = loader.get_ground_truth_stats()
    logger.info("=== KuaiRand Ground Truth ===")
    logger.info("  interactions:    %d", gt.n_interactions)
    logger.info("  like_rate:       %.4f", gt.like_rate)
    logger.info("  comment_rate:    %.4f", gt.comment_rate)
    logger.info("  forward_rate:    %.4f", gt.forward_rate)
    logger.info("  follow_rate:     %.4f", gt.follow_rate)
    logger.info("  skip_rate(<2s):  %.4f", gt.skip_rate_2s)
    logger.info("  completion(>80%%): %.4f", gt.completion_rate_80)
    logger.info("  avg_watch_ratio: %.4f", gt.avg_watch_ratio)

    logger.info("Building personas from real user data...")
    personas = loader.build_personas(n_users=20, seed=42)
    logger.info("Built %d personas", len(personas))

    logger.info("Building content pool from real videos...")
    content_pool = loader.build_content_pool_from_data()
    logger.info("Content pool: %d videos", len(content_pool.videos))

    config = SimulationConfig(
        num_agents=len(personas),
        num_simulated_days=1,
        max_videos_per_session=15,
        max_sessions_per_day=1,
        memory_window=30,
        seed=42,
        rec_top_k=10,
        rec_epsilon=0.15,
        rec_diversity_weight=0.3,
    )

    scheduler = LLMScheduler(
        provider="deepseek",
        model="deepseek-chat",
        temperature=0.4,
        max_tokens=300,
        max_concurrent=32,
        timeout_seconds=60,
    )

    engine = SimulationEngine(config, scheduler)
    engine.content_pool = content_pool
    from src.recommendation.surrogate_recsys import SurrogateRecSys
    engine.recsys = SurrogateRecSys(
        content_pool=content_pool,
        top_k=config.rec_top_k,
        epsilon=config.rec_epsilon,
        diversity_weight=config.rec_diversity_weight,
        seed=config.seed,
    )
    engine.initialize_agents(personas=personas)

    logger.info("Running simulation with %d agents...", len(personas))
    start = time.time()
    records = await engine.run()
    elapsed = time.time() - start
    logger.info("Simulation done in %.1fs, %d interactions", elapsed, len(records))

    if not records:
        logger.error("No interactions recorded!")
        await scheduler.close()
        return

    sim_wr = np.array([r.watch_ratio for r in records])
    sim_like = sum(1 for r in records if r.liked) / len(records)
    sim_comment = sum(1 for r in records if r.commented) / len(records)
    sim_share = sum(1 for r in records if r.shared) / len(records)
    sim_skip = sum(1 for r in records if r.watch_ratio < 0.15) / len(records)
    sim_complete = sum(1 for r in records if r.watch_ratio >= 0.8) / len(records)

    logger.info("=== Simulation Results ===")
    logger.info("  interactions:    %d", len(records))
    logger.info("  like_rate:       %.4f (real: %.4f)", sim_like, gt.like_rate)
    logger.info("  comment_rate:    %.4f (real: %.4f)", sim_comment, gt.comment_rate)
    logger.info("  share_rate:      %.4f (real: %.4f)", sim_share, gt.forward_rate)
    logger.info("  skip_rate:       %.4f (real: %.4f)", sim_skip, gt.skip_rate_2s)
    logger.info("  completion_rate: %.4f (real: %.4f)", sim_complete, gt.completion_rate_80)
    logger.info("  avg_watch_ratio: %.4f (real: %.4f)", sim_wr.mean(), gt.avg_watch_ratio)

    real_wr = loader.get_real_watch_ratios()
    wr_comparison = compare_distributions(sim_wr, real_wr, label="watch_ratio")
    logger.info("=== Distribution Comparison (watch_ratio) ===")
    logger.info("  Wasserstein:   %.4f", wr_comparison["wasserstein"])
    logger.info("  JS divergence: %.4f", wr_comparison["js_divergence"])
    logger.info("  KS statistic:  %.4f (p=%.4f)", wr_comparison["ks_statistic"], wr_comparison["ks_p_value"])
    logger.info("  Mean diff:     %.1f%%", wr_comparison["mean_diff_pct"])

    report = {
        "ground_truth": {
            "n_interactions": gt.n_interactions,
            "like_rate": gt.like_rate,
            "comment_rate": gt.comment_rate,
            "forward_rate": gt.forward_rate,
            "skip_rate_2s": gt.skip_rate_2s,
            "completion_rate_80": gt.completion_rate_80,
            "avg_watch_ratio": gt.avg_watch_ratio,
        },
        "simulation": {
            "n_interactions": len(records),
            "n_agents": len(personas),
            "like_rate": sim_like,
            "comment_rate": sim_comment,
            "share_rate": sim_share,
            "skip_rate": sim_skip,
            "completion_rate": sim_complete,
            "avg_watch_ratio": float(sim_wr.mean()),
        },
        "distribution_comparison": wr_comparison,
        "elapsed_seconds": elapsed,
        "llm_stats": scheduler.get_stats(),
    }

    with open(os.path.join(output_dir, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Report saved to %s", output_dir)
    await scheduler.close()


if __name__ == "__main__":
    asyncio.run(main())
