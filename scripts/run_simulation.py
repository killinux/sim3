#!/usr/bin/env python3
"""Run the user simulation system."""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.metrics import compute_metrics, compute_per_user_metrics
from src.analysis.treatment_effect import compute_multiple_effects
from src.simulation.engine import SimulationConfig, SimulationEngine
from src.simulation.scheduler import LLMScheduler
from src.validation.aa_test import run_aa_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_simulation_config(raw: dict) -> SimulationConfig:
    sim = raw.get("simulation", {})
    agent = raw.get("agent", {})
    rec = raw.get("recommendation", {})
    ab = raw.get("ab_test", {})

    return SimulationConfig(
        num_agents=sim.get("num_agents", 100),
        num_simulated_days=sim.get("num_simulated_days", 1),
        max_videos_per_session=agent.get("max_videos_per_session", 50),
        max_sessions_per_day=agent.get("max_sessions_per_day", 5),
        memory_window=agent.get("memory_window", 30),
        seed=sim.get("random_seed", 42),
        content_pool_size=raw.get("content_pool", {}).get("num_videos", 5000),
        num_categories=raw.get("content_pool", {}).get("num_categories", 20),
        rec_top_k=rec.get("top_k", 10),
        rec_epsilon=rec.get("epsilon", 0.1),
        rec_diversity_weight=rec.get("diversity_weight", 0.2),
        ab_test_enabled=ab.get("enabled", False),
        ab_treatment_ratio=ab.get("treatment_ratio", 0.5),
        ab_treatment_config=ab.get("treatment_config", {}),
    )


async def main(config_path: str, output_dir: str, run_aa: bool = False, override_agents=None):
    raw_config = load_config(config_path)
    if override_agents is not None:
        raw_config.setdefault("simulation", {})["num_agents"] = override_agents
    sim_config = build_simulation_config(raw_config)
    llm_config = raw_config.get("llm", {})

    logger.info("Configuration loaded: %d agents, %d days", sim_config.num_agents, sim_config.num_simulated_days)

    scheduler = LLMScheduler(
        provider=llm_config.get("provider"),
        model=llm_config.get("model", "gpt-4o-mini"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 300),
        max_concurrent=llm_config.get("max_concurrent", 64),
        timeout_seconds=llm_config.get("timeout_seconds", 30.0),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
    )

    engine = SimulationEngine(sim_config, scheduler)
    engine.initialize_agents()

    logger.info("Starting simulation...")
    start_time = time.time()
    records = await engine.run()
    elapsed = time.time() - start_time
    logger.info("Simulation complete in %.1f seconds", elapsed)

    os.makedirs(output_dir, exist_ok=True)

    summary = engine.get_interaction_summary()
    summary["elapsed_seconds"] = elapsed
    logger.info("Summary: %s", json.dumps(summary, indent=2, default=str))

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    metrics = compute_metrics(records)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics by variant: %s", json.dumps(metrics, indent=2, default=str))

    if sim_config.ab_test_enabled:
        effects = compute_multiple_effects(records)
        with open(os.path.join(output_dir, "treatment_effects.json"), "w") as f:
            json.dump(effects, f, indent=2, default=str)
        logger.info("Treatment effects:")
        for metric_name, effect in effects.items():
            sig = "***" if effect.get("significant_bh") else ""
            logger.info(
                "  %s: effect=%.4f, p=%.4f %s",
                metric_name,
                effect.get("relative_effect", 0),
                effect.get("p_value", 1),
                sig,
            )

    if run_aa:
        logger.info("Running A/A validation (100 runs)...")
        aa_result = run_aa_validation(records, n_runs=100)
        with open(os.path.join(output_dir, "aa_validation.json"), "w") as f:
            json.dump(aa_result, f, indent=2, default=str)
        logger.info("A/A validation: %s", aa_result["verdict"])

    user_metrics = compute_per_user_metrics(records)
    with open(os.path.join(output_dir, "user_metrics.json"), "w") as f:
        json.dump(user_metrics, f, indent=2, default=str)

    await scheduler.close()
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run user simulation")
    parser.add_argument(
        "--config", default="configs/simulation.yaml", help="Config file path"
    )
    parser.add_argument(
        "--output", default="output/run", help="Output directory"
    )
    parser.add_argument(
        "--aa-test", action="store_true", help="Run A/A validation"
    )
    parser.add_argument(
        "--agents", type=int, default=None, help="Override num_agents from config"
    )
    args = parser.parse_args()
    asyncio.run(main(args.config, args.output, run_aa=args.aa_test, override_agents=args.agents))
