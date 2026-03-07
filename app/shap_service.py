from __future__ import annotations

import math

from app.models import ShapScore


def _to_float(value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def compute_shap(features: dict[str, float | int | bool]) -> tuple[float, list[ShapScore]]:
    if not features:
        return 0.0, []

    items = [(k, _to_float(v)) for k, v in features.items()]
    # Lightweight proxy contribution scoring for local explainability endpoint.
    scale = max(1.0, sum(abs(v) for _, v in items))
    scores = []
    for key, value in items:
        contrib = value / scale
        scores.append(ShapScore(feature=key, contribution=contrib))

    scores.sort(key=lambda x: abs(x.contribution), reverse=True)
    # Stable base value in [0,1] from signed mean.
    mean_contrib = sum(s.contribution for s in scores) / max(1, len(scores))
    base_value = 1.0 / (1.0 + math.exp(-mean_contrib))
    return base_value, scores[:12]
