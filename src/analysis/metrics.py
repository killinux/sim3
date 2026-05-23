from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.simulation.engine import InteractionRecord


def compute_metrics(records: list[InteractionRecord]) -> dict:
    if not records:
        return {}

    by_variant: dict[str, list[InteractionRecord]] = defaultdict(list)
    for r in records:
        by_variant[r.variant].append(r)

    result = {}
    for variant, variant_records in by_variant.items():
        n = len(variant_records)
        watch_ratios = [r.watch_ratio for r in variant_records]
        result[variant] = {
            "n_interactions": n,
            "n_users": len({r.user_id for r in variant_records}),
            "avg_watch_ratio": np.mean(watch_ratios),
            "median_watch_ratio": np.median(watch_ratios),
            "std_watch_ratio": np.std(watch_ratios),
            "like_rate": sum(1 for r in variant_records if r.liked) / n,
            "comment_rate": sum(1 for r in variant_records if r.commented) / n,
            "share_rate": sum(1 for r in variant_records if r.shared) / n,
            "follow_rate": sum(1 for r in variant_records if r.followed) / n,
            "skip_rate": sum(1 for r in variant_records if r.watch_ratio < 0.15) / n,
            "completion_rate": sum(1 for r in variant_records if r.watch_ratio >= 0.8) / n,
            "avg_videos_per_session": n / max(1, len({r.session_id for r in variant_records})),
            "watch_ratio_distribution": {
                "p10": np.percentile(watch_ratios, 10),
                "p25": np.percentile(watch_ratios, 25),
                "p50": np.percentile(watch_ratios, 50),
                "p75": np.percentile(watch_ratios, 75),
                "p90": np.percentile(watch_ratios, 90),
            },
            "category_distribution": _category_distribution(variant_records),
        }

    return result


def compute_per_user_metrics(records: list[InteractionRecord]) -> dict[str, dict]:
    by_user: dict[str, list[InteractionRecord]] = defaultdict(list)
    for r in records:
        by_user[r.user_id].append(r)

    user_metrics = {}
    for uid, user_records in by_user.items():
        n = len(user_records)
        user_metrics[uid] = {
            "n_interactions": n,
            "n_sessions": len({r.session_id for r in user_records}),
            "avg_watch_ratio": np.mean([r.watch_ratio for r in user_records]),
            "like_rate": sum(1 for r in user_records if r.liked) / n,
            "skip_rate": sum(1 for r in user_records if r.watch_ratio < 0.15) / n,
            "variant": user_records[0].variant,
        }
    return user_metrics


def _category_distribution(records: list[InteractionRecord]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.category] += 1
    total = sum(counts.values())
    return {cat: count / total for cat, count in sorted(counts.items(), key=lambda x: -x[1])}
