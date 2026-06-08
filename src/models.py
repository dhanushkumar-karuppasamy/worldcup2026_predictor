# src/models.py
# ─────────────────────────────────────────────────────────────
# Three model wrappers used throughout the project:
#   1. BalancedLR   — Logistic Regression, handles class imbalance
#   2. WCXGBoost    — XGBoost, best classifier overall
#   3. PoissonGoals — Predicts scorelines, derives win/draw/loss probs
#
# Each exposes: .fit(X, y) / .predict(X) / .predict_proba(X)
# PoissonGoals additionally exposes: .predict_scoreline(X)
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import poisson

from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, classification_report
)
from xgboost import XGBClassifier

# Label convention: -1=Away Win, 0=Draw, 1=Home Win
LABELS   = [-1, 0, 1]
LABEL_NAMES = ["Away Win", "Draw", "Home Win"]

# Features used specifically by the Poisson goals model (subset of all features)
POISSON_FEATS = [
    "delta_elo", "delta_pts_5", "delta_gd_5",
    "is_knockout", "home_is_host", "away_is_host", "delta_wpts_5"
]


# ════════════════════════════════════════════════════════════
# 1. BALANCED LOGISTIC REGRESSION
# ════════════════════════════════════════════════════════════

class BalancedLR:
    """
    Multinomial Logistic Regression with class balancing and median imputation.
    Good baseline — well-calibrated probabilities.
    """
    def __init__(self, C: float = 0.5):
        self.C       = C
        self.model   = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=C,
                solver="lbfgs"
            )),
        ])
        self.classes_ = np.array(LABELS)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.named_steps["clf"].classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def class_index(self, label: int) -> int:
        return list(self.classes_).index(label)


# ════════════════════════════════════════════════════════════
# 2. XGBOOST CLASSIFIER
# ════════════════════════════════════════════════════════════

class WCXGBoost:
    """
    XGBoost 3-class classifier. Best overall accuracy.
    Uses LabelEncoder internally (XGBoost needs 0,1,2 labels).
    """
    def __init__(self, n_estimators: int = 150, max_depth: int = 3,
                 learning_rate: float = 0.05):
        self.le      = LabelEncoder()
        self.imputer = SimpleImputer(strategy="median")
        self.model   = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )
        self.le.fit(LABELS)
        self.classes_ = np.array(LABELS)

    def fit(self, X, y, X_val=None, y_val=None):
        X_imp = self.imputer.fit_transform(X)
        y_enc = self.le.transform(y)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_imp = self.imputer.transform(X_val)
            y_val_enc = self.le.transform(y_val)
            eval_set  = [(X_val_imp, y_val_enc)]

        self.model.fit(X_imp, y_enc, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X):
        X_imp = self.imputer.transform(X)
        return self.le.inverse_transform(self.model.predict(X_imp))

    def predict_proba(self, X):
        X_imp = self.imputer.transform(X)
        return self.model.predict_proba(X_imp)

    def feature_importances(self, feature_names: list) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)


# ════════════════════════════════════════════════════════════
# 3. POISSON GOALS MODEL
# ════════════════════════════════════════════════════════════

class PoissonGoals:
    """
    Predicts expected goals (λ_home, λ_away) using Poisson regression,
    then derives match probabilities by integrating over the score distribution.

    This is the strongest accuracy model but only outputs hard predictions
    (no log loss). Best used in the tournament simulator for scoreline simulation.
    """
    def __init__(self, alpha: float = 0.1):
        self.imputer     = SimpleImputer(strategy="median")
        self.model_home  = PoissonRegressor(alpha=alpha, max_iter=2000)
        self.model_away  = PoissonRegressor(alpha=alpha, max_iter=2000)
        self.feat_cols   = None
        self.classes_    = np.array(LABELS)

    def _get_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Use only the Poisson-relevant features that exist in X."""
        cols = [c for c in POISSON_FEATS if c in X.columns]
        self.feat_cols = cols
        return X[cols]

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        Xp    = self._get_features(X)
        X_imp = self.imputer.fit_transform(Xp)
        self.model_home.fit(X_imp, y_home)
        self.model_away.fit(X_imp, y_away)
        return self

    def predict_lambda(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lambda_home, lambda_away) — expected goals for each team."""
        Xp    = X[[c for c in self.feat_cols if c in X.columns]]
        X_imp = self.imputer.transform(Xp)
        lh    = np.maximum(0.05, self.model_home.predict(X_imp))
        la    = np.maximum(0.05, self.model_away.predict(X_imp))
        return lh, la

    @staticmethod
    def match_probs(lam_h: float, lam_a: float, max_goals: int = 10) -> tuple[float, float, float]:
        """
        Integrates Poisson PMF over all scorelines up to max_goals × max_goals.
        Returns (p_home_win, p_draw, p_away_win).
        """
        ph = pd_ = pa = 0.0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
                if   h > a: ph  += p
                elif h < a: pa  += p
                else:       pd_ += p
        return ph, pd_, pa

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns (N, 3) array of [P(away), P(draw), P(home)] — same order as LABELS."""
        lh, la = self.predict_lambda(X)
        rows   = [self.match_probs(h, a) for h, a in zip(lh, la)]
        # rows are (ph, pd, pa) → reorder to match LABELS = [-1, 0, 1]
        return np.array([[pa, pd_, ph] for ph, pd_, pa in rows])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        idx   = np.argmax(proba, axis=1)
        return np.array(LABELS)[idx]

    def simulate_scoreline(self, lam_h: float, lam_a: float,
                           max_goals: int = 8) -> tuple[int, int]:
        """
        Randomly samples a scoreline from the Poisson distribution.
        Used by the Monte Carlo simulator.
        """
        return (
            np.random.poisson(lam_h),
            np.random.poisson(lam_a)
        )


# ════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════

def evaluate_model(name: str, y_true, y_pred, y_proba, class_labels) -> dict:
    """Prints and returns a metrics dict for one model."""
    acc   = accuracy_score(y_true, y_pred)
    ll    = log_loss(y_true, y_proba, labels=class_labels)
    brier = np.mean([
        brier_score_loss((y_true == c).astype(int), y_proba[:, i])
        for i, c in enumerate(class_labels)
    ])
    print(f"\n{'='*58}")
    print(f"  {name}")
    print(f"{'='*58}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Log Loss  : {ll:.4f}")
    print(f"  Brier     : {brier:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=LABEL_NAMES, zero_division=0))
    return {"name": name, "accuracy": acc, "log_loss": ll, "brier": brier}


# ════════════════════════════════════════════════════════════
# SAVE / LOAD
# ════════════════════════════════════════════════════════════

def save_models(lr, xgb, poisson, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr,      path / "lr_model.pkl")
    joblib.dump(xgb,     path / "xgb_model.pkl")
    joblib.dump(poisson, path / "poisson_model.pkl")
    print(f"  Models saved to {path}/")


def load_models(path: Path) -> tuple:
    lr      = joblib.load(path / "lr_model.pkl")
    xgb     = joblib.load(path / "xgb_model.pkl")
    poisson = joblib.load(path / "poisson_model.pkl")
    print(f"  Models loaded from {path}/")
    return lr, xgb, poisson