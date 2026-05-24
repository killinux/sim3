#!/usr/bin/env python3
"""Validate that simulation correctly predicts AB test direction.

Uses KuaiRand's random vs standard recommendation as ground truth:
  - Random exposure → lower engagement (more skips, less completion)
  - Personalized recommendation → higher engagement

We simulate this by:
  - Control: high-randomness recommendation (epsilon=0.8, diversity=0.8)
  - Treatment: personalized recommendation (epsilon=0.05, diversity=0.1)

Success = simulation predicts the same direction as real data for all key metrics.
"""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REAL_DIRECTIONS = {
    "avg_watch_ratio": "up",
    "skip_rate": "down",
    "completion_rate": "up",
    "like_rate": "up",
}


def compute_variant_metrics(records, variant):
    subset = [r for r in records if r.variant == variant]
    if not subset:
        return {}
    n = len(subset)
    return {
        "n": n,
        "avg_watch_ratio": sum(r.watch_ratio for r in subset) / n,
        "skip_rate": sum(1 for r in subset if r.watch_ratio < 0.15) / n,
        "completion_rate": sum(1 for r in subset if r.watch_ratio >= 0.8) / n,
        "like_rate": sum(1 for r in subset if r.liked) / n,
    }


async def run_one(loader, scheduler, seed):
    personas = loader.build_personas(n_users=50, seed=seed)
    content_pool = loader.build_content_pool_from_data()

    config = SimulationConfig(
        num_agents=len(personas),
        num_simulated_days=1,
        max_videos_per_session=15,
        max_sessions_per_day=1,
        memory_window=30,
        seed=seed,
        rec_top_k=10,
        rec_epsilon=0.8,
        rec_diversity_weight=0.8,
        ab_test_enabled=True,
        ab_treatment_ratio=0.5,
        ab_treatment_config={
            "epsilon": 0.05,
            "diversity_weight": 0.1,
        },
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

    records = await engine.run()
    control = compute_variant_metrics(records, "control")
    treatment = compute_variant_metrics(records, "treatment")
    return control, treatment


async def main():
    output_dir = "output/ab_direction_validation"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading KuaiRand data...")
    loader = KuaiRandLoader("data/KuaiRand-Pure/data")

    scheduler = LLMScheduler(
        provider="deepseek",
        model="deepseek-chat",
        temperature=0.2,
        max_tokens=300,
        max_concurrent=32,
        timeout_seconds=60,
    )

    n_runs = 5
    all_controls = []
    all_treatments = []

    for i in range(n_runs):
        seed = 42 + i * 7
        logger.info("=== Run %d/%d (seed=%d) ===", i + 1, n_runs, seed)
        start = time.time()
        control, treatment = await run_one(loader, scheduler, seed)
        elapsed = time.time() - start
        all_controls.append(control)
        all_treatments.append(treatment)

        logger.info("  Run %d done in %.1fs", i + 1, elapsed)
        logger.info("  Control:   n=%d skip=%.1f%% compl=%.1f%% avg_wr=%.3f like=%.2f%%",
                     control["n"], control["skip_rate"]*100, control["completion_rate"]*100,
                     control["avg_watch_ratio"], control["like_rate"]*100)
        logger.info("  Treatment: n=%d skip=%.1f%% compl=%.1f%% avg_wr=%.3f like=%.2f%%",
                     treatment["n"], treatment["skip_rate"]*100, treatment["completion_rate"]*100,
                     treatment["avg_watch_ratio"], treatment["like_rate"]*100)

    logger.info("\n=== AB Direction Validation Results (%d runs) ===", n_runs)

    metrics = ["avg_watch_ratio", "skip_rate", "completion_rate", "like_rate"]
    results = {}

    for metric in metrics:
        ctrl_vals = [c[metric] for c in all_controls]
        treat_vals = [t[metric] for t in all_treatments]
        ctrl_mean = np.mean(ctrl_vals)
        treat_mean = np.mean(treat_vals)
        diff = treat_mean - ctrl_mean
        diff_pct = diff / max(ctrl_mean, 0.0001) * 100

        sim_direction = "up" if diff > 0 else "down"
        real_direction = REAL_DIRECTIONS[metric]
        match = sim_direction == real_direction

        n_correct = sum(1 for c, t in zip(ctrl_vals, treat_vals)
                        if (t > c) == (real_direction == "up"))

        results[metric] = {
            "control_mean": ctrl_mean,
            "treatment_mean": treat_mean,
            "diff": diff,
            "diff_pct": diff_pct,
            "sim_direction": sim_direction,
            "real_direction": real_direction,
            "direction_match": match,
            "n_correct_runs": n_correct,
            "n_total_runs": n_runs,
        }

        status = "✅ MATCH" if match else "❌ MISMATCH"
        logger.info("  %-20s ctrl=%.4f treat=%.4f diff=%+.1f%% real=%s sim=%s %s (%d/%d runs)",
                     metric, ctrl_mean, treat_mean, diff_pct,
                     real_direction, sim_direction, status, n_correct, n_runs)

    n_match = sum(1 for r in results.values() if r["direction_match"])
    n_total = len(results)
    all_match = n_match == n_total

    logger.info("\n=== Summary ===")
    logger.info("  Direction match: %d/%d metrics", n_match, n_total)
    logger.info("  Overall: %s", "✅ PASS" if all_match else "❌ FAIL")

    report = {
        "config": {
            "n_agents": 50,
            "n_runs": n_runs,
            "control": {"epsilon": 0.8, "diversity_weight": 0.8},
            "treatment": {"epsilon": 0.05, "diversity_weight": 0.1},
        },
        "real_directions": REAL_DIRECTIONS,
        "results": results,
        "summary": {
            "n_match": n_match,
            "n_total": n_total,
            "pass": all_match,
        },
        "per_run": {
            "controls": all_controls,
            "treatments": all_treatments,
        },
        "llm_stats": scheduler.get_stats(),
    }

    report_path = os.path.join(output_dir, "ab_direction_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    await scheduler.close()


if __name__ == "__main__":
    asyncio.run(main())
