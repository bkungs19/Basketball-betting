"""Odds math shared by the model and the parlay builder."""
import math


def american_to_decimal(american: float) -> float:
    american = float(american)
    if american >= 100:
        return 1 + american / 100
    if american <= -100:
        return 1 + 100 / -american
    raise ValueError(f"bad American odds: {american}")


def decimal_to_american(decimal: float) -> int:
    if decimal >= 2:
        return round((decimal - 1) * 100)
    return round(-100 / (decimal - 1))


def implied_prob(american: float) -> float:
    return 1 / american_to_decimal(american)


def no_vig_pair(over_american: float, under_american: float) -> tuple[float, float]:
    """Fair over/under probabilities from a two-way price (multiplicative devig)."""
    po, pu = implied_prob(over_american), implied_prob(under_american)
    total = po + pu
    return po / total, pu / total


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def blend(p_model: float, p_market: float | None, model_weight: float) -> float:
    """Blend model and market probabilities in log-odds space."""
    if p_market is None or math.isnan(p_market):
        return p_model
    z = model_weight * logit(p_model) + (1 - model_weight) * logit(p_market)
    return 1 / (1 + math.exp(-z))
