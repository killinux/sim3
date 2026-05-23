from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats

from src.simulation.engine import InteractionRecord


def compute_treatment_effect(
    records: list[InteractionRecord],
    metric_fn: callable | None = None,
) -> dict:
    if metric_fn is None:
        metric_fn = lambda r: r.watch_ratio

    by_variant_user: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_variant_user[r.variant][r.user_id].append(metric_fn(r))

    user_means: dict[str, list[float]] = {}
    for variant, users in by_variant_user.items():
        user_means[variant] = [np.mean(vals) for vals in users.values()]

    if "treatment" not in user_means or "control" not in user_means:
        return {"error": "Need both treatment and control groups"}

    treatment_vals = np.array(user_means["treatment"])
    control_vals = np.array(user_means["control"])

    treatment_mean = np.mean(treatment_vals)
    control_mean = np.mean(control_vals)
    absolute_effect = treatment_mean - control_mean
    relative_effect = absolute_effect / control_mean if control_mean != 0 else 0.0

    t_stat, p_value = stats.ttest_ind(treatment_vals, control_vals, equal_var=False)

    boot_effects = bootstrap_treatment_effect(treatment_vals, control_vals)
    ci_lower = np.percentile(boot_effects, 2.5)
    ci_upper = np.percentile(boot_effects, 97.5)

    pooled_std = np.sqrt(
        (np.var(treatment_vals) + np.var(control_vals)) / 2
    )
    cohens_d = absolute_effect / pooled_std if pooled_std > 0 else 0.0

    return {
        "treatment_mean": treatment_mean,
        "control_mean": control_mean,
        "absolute_effect": absolute_effect,
        "relative_effect": relative_effect,
        "t_statistic": t_stat,
        "p_value": p_value,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "cohens_d": cohens_d,
        "n_treatment": len(treatment_vals),
        "n_control": len(control_vals),
        "significant": p_value < 0.05,
    }


def bootstrap_treatment_effect(
    treatment: np.ndarray,
    control: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    effects = np.empty(n_bootstrap)
    n_t = len(treatment)
    n_c = len(control)
    for i in range(n_bootstrap):
        t_sample = treatment[rng.randint(0, n_t, size=n_t)]
        c_sample = control[rng.randint(0, n_c, size=n_c)]
        effects[i] = np.mean(t_sample) - np.mean(c_sample)
    return effects


def compute_multiple_effects(
    records: list[InteractionRecord],
) -> dict[str, dict]:
    metrics = {
        "watch_ratio": lambda r: r.watch_ratio,
        "liked": lambda r: float(r.liked),
        "commented": lambda r: float(r.commented),
        "shared": lambda r: float(r.shared),
        "completed": lambda r: float(r.watch_ratio >= 0.8),
        "skipped": lambda r: float(r.watch_ratio < 0.15),
    }

    results = {}
    p_values = []

    for name, fn in metrics.items():
        effect = compute_treatment_effect(records, metric_fn=fn)
        results[name] = effect
        if "p_value" in effect:
            p_values.append((name, effect["p_value"]))

    if p_values:
        corrected = benjamini_hochberg([p for _, p in p_values])
        for i, (name, _) in enumerate(p_values):
            results[name]["p_value_bh"] = corrected[i]
            results[name]["significant_bh"] = corrected[i] < 0.05

    return results


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    prev = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adjusted = p * n / rank
        corrected[orig_idx] = min(prev, min(adjusted, 1.0))
        prev = corrected[orig_idx]
    return corrected
