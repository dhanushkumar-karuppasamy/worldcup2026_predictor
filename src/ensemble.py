# src/ensemble.py
# ─────────────────────────────────────────────────────────────
# Ensemble model: weighted blend of XGBoost + Poisson probabilities.
#
# Why this works:
#   XGBoost   → best log loss / probability calibration
#   Poisson   → best raw accuracy (knows about scorelines)
#   Blend     → gets the benefits of both
#
# Tuned weights from backtest: XGBoost 40%, Poisson 60%
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pathlib import Path
from src.models import WCXGBoost, PoissonGoals, load_models, LABELS, LABEL_NAMES
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

MODEL_DIR = Path("models")


class EnsembleModel:
    """
    Weighted average of XGBoost and Poisson probability outputs.

    Both models output P(away win), P(draw), P(home win) in that order.
    The ensemble simply blends these with configurable weights.
    """

    def __init__(self, xgb_weight: float = 0.40, poisson_weight: float = 0.60):
        assert abs(xgb_weight + poisson_weight - 1.0) < 1e-6, "Weights must sum to 1"
        self.xgb_w    = xgb_weight
        self.pois_w   = poisson_weight
        self.xgb      = None
        self.poisson  = None
        self.classes_ = np.array(LABELS)  # [-1, 0, 1]

    def load(self, model_dir: Path = MODEL_DIR):
        """Load saved XGBoost and Poisson models."""
        _, self.xgb, self.poisson = load_models(model_dir)
        return self

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            home_score: pd.Series, away_score: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Train both models from scratch (for backtesting)."""
        from src.models import WCXGBoost, PoissonGoals
        self.xgb = WCXGBoost()
        self.xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        self.poisson = PoissonGoals()
        self.poisson.fit(X_train, home_score, away_score)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns (N, 3) blended probability array.
        Column order matches LABELS: [P(away), P(draw), P(home)]
        """
        if self.xgb is None or self.poisson is None:
            raise RuntimeError("Call .load() or .fit() before predict_proba()")

        # XGBoost probabilities — already in [-1, 0, 1] order
        p_xgb  = self.xgb.predict_proba(X)     # shape (N, 3)

        # Poisson probabilities — also in [-1, 0, 1] order
        p_pois = self.poisson.predict_proba(X)  # shape (N, 3)

        # Weighted blend
        blended = self.xgb_w * p_xgb + self.pois_w * p_pois

        # Re-normalise (floating point safety)
        blended = blended / blended.sum(axis=1, keepdims=True)
        return blended

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns hard label predictions: -1, 0, or 1."""
        proba = self.predict_proba(X)
        idx   = np.argmax(proba, axis=1)
        return np.array(LABELS)[idx]

    def tune_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[float, float]:
        """
        Grid-searches the best XGB/Poisson weight split on a validation set.
        Returns (best_xgb_weight, best_poisson_weight).
        """
        best_acc    = 0
        best_weights = (0.4, 0.6)

        p_xgb  = self.xgb.predict_proba(X_val)
        p_pois = self.poisson.predict_proba(X_val)

        for xgb_w in np.arange(0.1, 1.0, 0.05):
            pois_w = round(1.0 - xgb_w, 2)
            blend  = xgb_w * p_xgb + pois_w * p_pois
            blend  = blend / blend.sum(axis=1, keepdims=True)
            preds  = np.array(LABELS)[np.argmax(blend, axis=1)]
            acc    = accuracy_score(y_val, preds)
            if acc > best_acc:
                best_acc     = acc
                best_weights = (round(xgb_w, 2), round(pois_w, 2))

        print(f"  Best weights → XGB: {best_weights[0]}, Poisson: {best_weights[1]}  (val acc: {best_acc:.4f})")
        self.xgb_w  = best_weights[0]
        self.pois_w = best_weights[1]
        return best_weights


def evaluate_ensemble_backtest(wc: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    Leave-one-tournament-out backtest for the ensemble model.
    Compares ensemble accuracy vs individual models.
    Returns a summary DataFrame.
    """
    from sklearn.metrics import accuracy_score
    import warnings
    warnings.filterwarnings("ignore")

    test_years = [y for y in sorted(wc["year"].unique()) if y >= 1994]
    records    = []

    print("\n  Ensemble backtest (leave-one-out):")
    print(f"  {'Year':<6} {'XGB':>7} {'Poisson':>9} {'Ensemble':>10}")
    print("  " + "-" * 35)

    for test_yr in test_years:
        train = wc[(wc["year"] != test_yr) & wc["result_num"].notna()]
        test  = wc[wc["year"] == test_yr]
        if len(train) < 100 or len(test) < 10:
            continue

        X_tr, y_tr   = train[feat_cols], train["result_num"]
        X_te, y_te   = test[feat_cols],  test["result_num"]

        # Previous year as val
        all_yrs = sorted(wc["year"].unique())
        yr_idx  = list(all_yrs).index(test_yr)
        prev_yr = all_yrs[yr_idx - 1] if yr_idx > 0 else None
        val     = wc[wc["year"] == prev_yr] if prev_yr else train.tail(30)
        X_v, y_v = val[feat_cols], val["result_num"]

        ens = EnsembleModel()
        ens.fit(X_tr, y_tr, train["home_score"], train["away_score"], X_v, y_v)
        ens.tune_weights(X_v, y_v)

        acc_xgb  = accuracy_score(y_te, ens.xgb.predict(X_te))
        acc_pois = accuracy_score(y_te, ens.poisson.predict(X_te))
        acc_ens  = accuracy_score(y_te, ens.predict(X_te))

        records.append({
            "year":         test_yr,
            "acc_xgb":      round(acc_xgb,  4),
            "acc_poisson":  round(acc_pois, 4),
            "acc_ensemble": round(acc_ens,  4),
            "improvement":  round(acc_ens - max(acc_xgb, acc_pois), 4),
        })
        better = "↑" if acc_ens > max(acc_xgb, acc_pois) else ("=" if acc_ens == max(acc_xgb, acc_pois) else "↓")
        print(f"  {test_yr:<6} {acc_xgb:>7.3f} {acc_pois:>9.3f} {acc_ens:>9.3f}  {better}")

    df = pd.DataFrame(records)
    print(f"\n  Avg XGB      : {df.acc_xgb.mean():.4f}")
    print(f"  Avg Poisson  : {df.acc_poisson.mean():.4f}")
    print(f"  Avg Ensemble : {df.acc_ensemble.mean():.4f}  (avg improvement: {df.improvement.mean():+.4f})")
    return df