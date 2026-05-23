from __future__ import annotations

import numpy as np
from scipy import stats

from src.analysis.treatment_effect import compute_treatment_effect
from src.simulation.engine import InteractionRecord


def run_aa_validation(
    records: list[InteractionRecord],
    n_runs: int = 100,
    seed: int = 42,
) -> dict:
    rng = np.random.RandomState(seed)
    user_ids = list({r.user_id for r in records})

    p_values = []
    for _ in range(n_runs):
        shuffled_records = _random_ab_split(records, user_ids, rng)
        effect = compute_treatment_effect(shuffled_records)
        if "p_value" in effect:
            p_values.append(effect["p_value"])

    if not p_values:
        return {"error": "no valid p-values generated"}

    p_values = np.array(p_values)

    ks_stat, ks_p = stats.kstest(p_values, "uniform")

    false_positive_rate = np.mean(p_values < 0.05)

    return {
        "n_runs": len(p_values),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "p_values_uniform": ks_p > 0.05,
        "false_positive_rate": float(false_positive_rate),
        "fpr_acceptable": false_positive_rate < 0.10,
        "mean_p_value": float(np.mean(p_values)),
        "median_p_value": float(np.median(p_values)),
        "p_value_quantiles": {
            "p10": float(np.percentile(p_values, 10)),
            "p25": float(np.percentile(p_values, 25)),
            "p50": float(np.percentile(p_values, 50)),
            "p75": float(np.percentile(p_values, 75)),
            "p90": float(np.percentile(p_values, 90)),
        },
        "verdict": (
            "PASS" if (ks_p > 0.05 and false_positive_rate < 0.10) else "FAIL"
        ),
    }


def _random_ab_split(
    records: list[InteractionRecord],
    user_ids: list[str],
    rng: np.random.RandomState,
) -> list[InteractionRecord]:
    shuffled = list(user_ids)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    treatment_users = set(shuffled[:mid])

    result = []
    for r in records:
        new_variant = "treatment" if r.user_id in treatment_users else "control"
        result.append(InteractionRecord(
            user_id=r.user_id,
            video_id=r.video_id,
            category=r.category,
            duration_seconds=r.duration_seconds,
            watch_ratio=r.watch_ratio,
            liked=r.liked,
            commented=r.commented,
            shared=r.shared,
            followed=r.followed,
            session_id=r.session_id,
            position_in_session=r.position_in_session,
            fatigue_at_decision=r.fatigue_at_decision,
            variant=new_variant,
            simulated_day=r.simulated_day,
            timestamp=r.timestamp,
        ))
    return result
