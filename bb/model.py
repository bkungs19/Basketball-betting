"""Predict each player's stat distribution and the chance of going over a line.

For every market a gradient-boosted model predicts the expected stat, and a
second model predicts how spread out the outcome is. Together they give a
negative binomial distribution, so P(over 24.5 points) is one CDF lookup.
Probabilities are then calibrated on a held-out season.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from .config import DATA
from .features import TARGETS, feature_columns

MIN_DISPERSION = 1.05  # variance is at least 5% above the mean


def _mean_model():
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=80, l2_regularization=1.0, random_state=0,
    )


def _var_model():
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0,
    )


def p_over(mu, var, line):
    """P(X > line) for a negative binomial with the given mean and variance."""
    mu = np.maximum(np.asarray(mu, float), 0.05)
    var = np.maximum(np.asarray(var, float), mu * MIN_DISPERSION)
    k = np.floor(np.asarray(line, float))
    r = mu ** 2 / (var - mu)
    p = r / (r + mu)
    out = nbinom.sf(k, r, p)
    return np.where(np.isfinite(out), out, poisson.sf(k, mu))


def pseudo_lines(f: pd.DataFrame, col: str, rng: np.random.Generator) -> np.ndarray:
    """Book-like lines for backtesting: recent average rounded to a half point, nudged a little."""
    base = np.floor(f[f"ewm15_{col}"].fillna(f[f"ewm5_{col}"]).values) + 0.5
    jitter = rng.choice([-1, 0, 0, 1], size=len(base))
    return np.maximum(base + jitter, 0.5)


HOLDOUT_MIN_ROWS = 15000  # a full season has about 25,000 player-game rows


class PropModel:
    def __init__(self):
        self.features: list[str] = []
        self.mean: dict = {}
        self.var: dict = {}
        self.calib: dict = {}
        self.report: dict = {}

    def _var_X(self, X: pd.DataFrame, mu: np.ndarray, col: str) -> pd.DataFrame:
        V = X[[c for c in X.columns if col in c or c.startswith(("ewm5_min", "ewm15_min", "std10_min", "avg10_min"))]].copy()
        V["mu"] = mu
        return V

    def fit(self, f: pd.DataFrame, holdout_season: str | None = None) -> "PropModel":
        self.features = feature_columns(f)
        # Hold out the latest full season; the season under way joins only the
        # final refit until it's big enough (early on it has a handful of games).
        sizes = f.groupby("season").size()
        full = sorted(sizes[sizes >= HOLDOUT_MIN_ROWS].index)
        holdout_season = holdout_season or (full[-1] if full else sizes.index.max())
        train, hold = f[f["season"] < holdout_season], f[f["season"] == holdout_season]
        rng = np.random.default_rng(0)
        report = {"holdout_season": holdout_season, "train_rows": int(len(train)), "holdout_rows": int(len(hold)), "markets": {}}

        # The held-out season is split in time: the first half fits the spread
        # model and calibration, the second half is the untouched test set.
        cut = hold["game_date"].sort_values().iloc[len(hold) // 2]
        cal, test = hold[hold["game_date"] < cut], hold[hold["game_date"] >= cut]
        report["test_rows"] = int(len(test))
        report["test_from"] = str(pd.Timestamp(cut).date())
        rotation = (test["ewm15_min"] >= 20).values  # players who usually get prop lines

        for market, col in TARGETS.items():
            # 1) mean model fit on earlier seasons only
            m = _mean_model().fit(train[self.features], train[col].values)
            mu_cal = np.clip(m.predict(cal[self.features]), 0.05, None)
            mu_te = np.clip(m.predict(test[self.features]), 0.05, None)
            y_cal, y_te = cal[col].values, test[col].values

            # 2) spread model on the calibration half's squared errors
            v = _var_model().fit(self._var_X(cal[self.features], mu_cal, col), (y_cal - mu_cal) ** 2)
            var_cal = v.predict(self._var_X(cal[self.features], mu_cal, col))
            var_te = v.predict(self._var_X(test[self.features], mu_te, col))

            # 3) calibrate P(over) against book-like lines on the calibration half
            l_cal = pseudo_lines(cal, col, rng)
            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(
                p_over(mu_cal, var_cal, l_cal), (y_cal > l_cal).astype(float))

            # 4) score the untouched test half, rotation players only
            l_te = pseudo_lines(test, col, rng)
            p_te = iso.predict(p_over(mu_te, var_te, l_te))[rotation]
            hit = (y_te > l_te).astype(float)[rotation]
            yr, mur = y_te[rotation], mu_te[rotation]
            recent = test[f"ewm15_{col}"].fillna(0).values[rotation]
            mae_model = float(np.mean(np.abs(yr - mur)))
            mae_recent = float(np.mean(np.abs(yr - recent)))
            correct = np.where(p_te > 0.5, hit == 1, hit == 0)
            strong = np.abs(p_te - 0.5) >= 0.1
            report["markets"][market] = {
                "mae_model": round(mae_model, 3),
                "mae_recent_average": round(mae_recent, 3),
                "brier_model": round(float(np.mean((p_te - hit) ** 2)), 4),
                "brier_base_rate": round(float(np.mean((hit.mean() - hit) ** 2)), 4),
                "pick_accuracy": round(float(correct.mean()), 4),
                "strong_pick_accuracy": round(float(correct[strong].mean()), 4) if strong.any() else None,
                "strong_pick_share": round(float(strong.mean()), 4),
                "calibration": _calibration_table(p_te, hit),
            }
            self.var[market] = v
            self.calib[market] = iso

            # 5) refit the mean model on everything for live predictions
            self.mean[market] = _mean_model().fit(f[self.features], f[col].values)
            print(f"{market}: MAE {mae_model:.2f} vs recent-average {mae_recent:.2f}, "
                  f"pick accuracy {correct.mean():.3f}")
        self.report = report
        return self

    def predict(self, f: pd.DataFrame, market: str, lines) -> pd.DataFrame:
        col = TARGETS[market]
        X = f[self.features]
        mu = np.clip(self.mean[market].predict(X), 0.05, None)
        var = self.var[market].predict(self._var_X(X, mu, col))
        p_raw = p_over(mu, var, lines)
        p = self.calib[market].predict(p_raw)
        return pd.DataFrame({"proj": mu, "sd": np.sqrt(np.maximum(var, mu * MIN_DISPERSION)), "p_over": p}, index=f.index)

    def save_report(self):
        DATA.mkdir(exist_ok=True)
        (DATA / "model_report.json").write_text(json.dumps(self.report, indent=1))


def _calibration_table(p: np.ndarray, hit: np.ndarray) -> list[dict]:
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    out = []
    for b in range(10):
        m = idx == b
        if m.sum() >= 50:
            out.append({"predicted": round(float(p[m].mean()), 3), "actual": round(float(hit[m].mean()), 3), "n": int(m.sum())})
    return out
