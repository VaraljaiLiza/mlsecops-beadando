"""Döntés egy új modell aktívvá tételéről."""

import math


def should_promote(
    candidate_score: float,
    current_score: float | None,
    minimum_improvement: float = 0.0,
) -> bool:
    """Igazat ad, ha az új modell teljesíti a váltás feltételét."""
    if not math.isfinite(candidate_score) or not 0 <= candidate_score <= 1:
        raise ValueError("Az új modell pontszáma legyen 0 és 1 közötti szám.")

    if not math.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("Az elvárt javulás legyen véges, nemnegatív szám.")

    if current_score is None:
        return True

    if not math.isfinite(current_score) or not 0 <= current_score <= 1:
        raise ValueError("Az aktív modell pontszáma legyen 0 és 1 közötti szám.")

    return candidate_score >= current_score + minimum_improvement
