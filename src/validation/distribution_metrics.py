from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


def wasserstein_distance(simulated: np.ndarray, real: np.ndarray) -> float:
    return float(stats.wasserstein_distance(simulated, real))


def js_divergence(simulated: np.ndarray, real: np.ndarray, bins: int = 50) -> float:
    min_val = min(simulated.min(), real.min())
    max_val = max(simulated.max(), real.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    hist_sim, _ = np.histogram(simulated, bins=bin_edges, density=True)
    hist_real, _ = np.histogram(real, bins=bin_edges, density=True)

    hist_sim = hist_sim + 1e-10
    hist_real = hist_real + 1e-10
    hist_sim = hist_sim / hist_sim.sum()
    hist_real = hist_real / hist_real.sum()

    return float(jensenshannon(hist_sim, hist_real) ** 2)


def ks_test(simulated: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    stat, p_value = stats.ks_2samp(simulated, real)
    return float(stat), float(p_value)


def categorical_js_divergence(
    simulated_counts: dict[str, int],
    real_counts: dict[str, int],
) -> float:
    all_keys = sorted(set(simulated_counts.keys()) | set(real_counts.keys()))
    sim_vec = np.array([simulated_counts.get(k, 0) for k in all_keys], dtype=float)
    real_vec = np.array([real_counts.get(k, 0) for k in all_keys], dtype=float)

    sim_vec = sim_vec + 1e-10
    real_vec = real_vec + 1e-10
    sim_vec = sim_vec / sim_vec.sum()
    real_vec = real_vec / real_vec.sum()

    return float(jensenshannon(sim_vec, real_vec) ** 2)


def compare_distributions(
    simulated: np.ndarray,
    real: np.ndarray,
    label: str = "metric",
) -> dict:
    w_dist = wasserstein_distance(simulated, real)
    js_div = js_divergence(simulated, real)
    ks_stat, ks_p = ks_test(simulated, real)

    return {
        "label": label,
        "wasserstein": w_dist,
        "js_divergence": js_div,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "simulated_mean": float(np.mean(simulated)),
        "real_mean": float(np.mean(real)),
        "simulated_std": float(np.std(simulated)),
        "real_std": float(np.std(real)),
        "mean_diff_pct": float(
            abs(np.mean(simulated) - np.mean(real))
            / max(abs(np.mean(real)), 1e-10)
            * 100
        ),
    }


def full_fidelity_report(
    simulated_records: list[dict],
    real_records: list[dict],
    metrics_to_compare: list[str] | None = None,
) -> dict:
    if metrics_to_compare is None:
        metrics_to_compare = ["watch_ratio"]

    report = {}
    for metric in metrics_to_compare:
        sim_vals = np.array([r[metric] for r in simulated_records if metric in r])
        real_vals = np.array([r[metric] for r in real_records if metric in r])

        if len(sim_vals) == 0 or len(real_vals) == 0:
            report[metric] = {"error": "insufficient data"}
            continue

        report[metric] = compare_distributions(sim_vals, real_vals, label=metric)

    return report
