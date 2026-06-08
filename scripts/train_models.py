# scripts/train_models.py
# ─────────────────────────────────────────────────────────────
# Trains all 3 models and evaluates them two ways:
#
#   A) Standard split: train pre-2018, val 2018, test 2022
#   B) Leave-one-out backtest: test on every WC from 1994–2022
#      This answers: "how would my model have done on each WC?"
#
# Usage (from project root):
#   python scripts/train_models.py
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score

from src.features import get_features, FEATURE_COLS
from src.models import (
    BalancedLR, WCXGBoost, PoissonGoals,
    evaluate_model, save_models, LABELS, LABEL_NAMES
)

PROC_DIR   = Path("data/processed")
OUTPUT_DIR = Path("outputs")
MODEL_DIR  = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════

def load_features() -> pd.DataFrame:
    path = PROC_DIR / "wc_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "wc_features.parquet not found. Run: python scripts/build_pipeline.py"
        )
    wc = pd.read_parquet(path)
    print(f"Loaded feature matrix: {len(wc)} matches, {wc.shape[1]} columns.")
    return wc


# ════════════════════════════════════════════════════════════
# A) STANDARD TRAIN / VAL / TEST
# ════════════════════════════════════════════════════════════

def run_standard_evaluation(wc: pd.DataFrame, feat_cols: list) -> dict:
    """Train on pre-2018, validate on 2018, final test on 2022."""
    print("\n" + "="*58)
    print("  A) STANDARD SPLIT EVALUATION")
    print("="*58)

    train = wc[(wc["year"] < 2018) & wc["result_num"].notna()]
    val   = wc[wc["year"] == 2018]
    test  = wc[wc["year"] == 2022]

    X_train, y_train = train[feat_cols], train["result_num"]
    X_val,   y_val   = val[feat_cols],   val["result_num"]
    X_test,  y_test  = test[feat_cols],  test["result_num"]

    print(f"  Train: {len(X_train)} | Val (2018): {len(X_val)} | Test (2022): {len(X_test)}")

    # ── Train models ─────────────────────────────────────────
    lr  = BalancedLR()
    lr.fit(X_train, y_train)

    xgb = WCXGBoost()
    xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    pois = PoissonGoals()
    pois.fit(
        train[feat_cols],
        train["home_score"],
        train["away_score"]
    )

    # ── Evaluate on TEST (2022) ───────────────────────────────
    print("\n  --- 2022 WORLD CUP TEST RESULTS ---")
    proba_lr  = lr.predict_proba(X_test)
    pred_lr   = lr.predict(X_test)
    m1 = evaluate_model("Logistic Regression (2022)", y_test, pred_lr, proba_lr, lr.classes_)

    proba_xgb = xgb.predict_proba(X_test)
    pred_xgb  = xgb.predict(X_test)
    m2 = evaluate_model("XGBoost (2022)", y_test, pred_xgb, proba_xgb, xgb.classes_)

    pred_pois = pois.predict(X_test)
    p_acc     = accuracy_score(y_test, pred_pois)
    lh, la    = pois.predict_lambda(X_test)
    m3 = {"name": "Poisson Goals (2022)", "accuracy": p_acc, "log_loss": None, "brier": None}
    print(f"\n  Poisson Goals (2022): Accuracy={p_acc:.4f} | Avg λ_home={lh.mean():.2f} | Avg λ_away={la.mean():.2f}")

    # ── Per-match probability table ───────────────────────────
    prob_df = test[["date", "home_team", "away_team", "home_score", "away_score", "result_num"]].copy().reset_index(drop=True)
    cl = list(lr.classes_)
    prob_df["p_home"] = proba_lr[:, cl.index(1)]
    prob_df["p_draw"] = proba_lr[:, cl.index(0)]
    prob_df["p_away"] = proba_lr[:, cl.index(-1)]
    prob_df["pred_lr"]  = pred_lr
    prob_df["pred_xgb"] = pred_xgb
    prob_df["pred_pois"]= pred_pois
    prob_df["correct_lr"]  = (pred_lr  == y_test.values)
    prob_df["correct_xgb"] = (pred_xgb == y_test.values)
    prob_df.to_csv(OUTPUT_DIR / "predictions_2022.csv", index=False)
    print(f"\n  Per-match predictions saved → outputs/predictions_2022.csv")

    # ── Feature importance ────────────────────────────────────
    fi = xgb.feature_importances(feat_cols).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    fi.plot(kind="barh", ax=ax, color="#0F6E56")
    ax.set_title("Top 20 Feature Importances — XGBoost (trained pre-2018)")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Feature importance chart → outputs/feature_importance.png")

    # ── Calibration plot ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Calibration — 2022 World Cup Test Set", fontsize=13)
    for ax, outcome, oname in zip(axes, [-1, 0, 1], LABEL_NAMES):
        for mname, proba, classes in [
            ("LR",  proba_lr,  lr.classes_),
            ("XGB", proba_xgb, xgb.classes_),
        ]:
            cl2 = list(classes)
            if outcome not in cl2: continue
            by = (y_test.values == outcome).astype(int)
            pc = proba[:, cl2.index(outcome)]
            if by.sum() < 3: continue
            fp, mp2 = calibration_curve(by, pc, n_bins=5)
            ax.plot(mp2, fp, marker="o", label=mname)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        ax.set_title(oname)
        ax.set_xlabel("Predicted prob")
        ax.set_ylabel("Actual freq")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_2022.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  Calibration chart → outputs/calibration_2022.png")

    return {"lr": lr, "xgb": xgb, "poisson": pois, "metrics": [m1, m2, m3]}


# ════════════════════════════════════════════════════════════
# B) LEAVE-ONE-TOURNAMENT-OUT BACKTEST
# ════════════════════════════════════════════════════════════

def run_backtest(wc: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    Your idea, implemented properly.
    For each World Cup from 1994 to 2022:
      - Train on ALL other WC years
      - Test on the held-out year
      - Record accuracy for all 3 models

    This shows how the model performs across different eras.
    1994 is the lower bound because earlier WCs have too few prior matches
    for rolling features to be meaningful.
    """
    print("\n" + "="*58)
    print("  B) LEAVE-ONE-TOURNAMENT-OUT BACKTEST")
    print("="*58)

    test_years = [y for y in sorted(wc["year"].unique()) if y >= 1994]
    records    = []

    for test_yr in test_years:
        # Train on everything EXCEPT this year
        train = wc[(wc["year"] != test_yr) & wc["result_num"].notna()]
        test  = wc[wc["year"] == test_yr]

        if len(train) < 100 or len(test) < 10:
            continue

        X_train, y_train = train[feat_cols], train["result_num"]
        X_test,  y_test  = test[feat_cols],  test["result_num"]

        # Use previous year as val for XGBoost early stopping
        all_years  = sorted(wc["year"].unique())
        yr_idx     = list(all_years).index(test_yr)
        prev_yr    = all_years[yr_idx - 1] if yr_idx > 0 else None
        val        = wc[wc["year"] == prev_yr] if prev_yr else train.tail(30)
        X_val_bt   = val[feat_cols]
        y_val_bt   = val["result_num"]

        # Train 3 models
        lr_bt  = BalancedLR()
        lr_bt.fit(X_train, y_train)

        xgb_bt = WCXGBoost()
        xgb_bt.fit(X_train, y_train, X_val=X_val_bt, y_val=y_val_bt)

        pois_bt = PoissonGoals()
        pois_bt.fit(train[feat_cols], train["home_score"], train["away_score"])

        # Evaluate
        acc_lr   = accuracy_score(y_test, lr_bt.predict(X_test))
        acc_xgb  = accuracy_score(y_test, xgb_bt.predict(X_test))
        acc_pois = accuracy_score(y_test, pois_bt.predict(X_test))

        records.append({
            "year":       test_yr,
            "n_matches":  len(test),
            "acc_lr":     round(acc_lr,   4),
            "acc_xgb":    round(acc_xgb,  4),
            "acc_poisson":round(acc_pois, 4),
        })
        print(f"  {test_yr} ({len(test):2d} matches) | "
              f"LR={acc_lr:.3f}  XGB={acc_xgb:.3f}  Poisson={acc_pois:.3f}")

    backtest_df = pd.DataFrame(records)

    # ── Summary stats ─────────────────────────────────────────
    print(f"\n  Backtest summary ({len(test_years)} tournaments):")
    print(f"  LR avg accuracy    : {backtest_df.acc_lr.mean():.4f}  ± {backtest_df.acc_lr.std():.4f}")
    print(f"  XGB avg accuracy   : {backtest_df.acc_xgb.mean():.4f}  ± {backtest_df.acc_xgb.std():.4f}")
    print(f"  Poisson avg accuracy: {backtest_df.acc_poisson.mean():.4f}  ± {backtest_df.acc_poisson.std():.4f}")

    # ── Backtest chart ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x = backtest_df["year"]
    ax.plot(x, backtest_df["acc_lr"],      "o-", label="Logistic Regression", color="#3B8BD4")
    ax.plot(x, backtest_df["acc_xgb"],     "s-", label="XGBoost",             color="#0F6E56")
    ax.plot(x, backtest_df["acc_poisson"], "^-", label="Poisson Goals",       color="#BA7517")
    ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.5, label="Random baseline (33%)")
    ax.axhline(y=0.50,  color="lightgray", linestyle=":",  alpha=0.5)
    ax.set_xlabel("World Cup Year")
    ax.set_ylabel("Accuracy")
    ax.set_title("Leave-One-Tournament-Out Backtest Accuracy (1994–2022)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "backtest_accuracy.png", dpi=130, bbox_inches="tight")
    plt.close()

    backtest_df.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)
    print("  Backtest chart  → outputs/backtest_accuracy.png")
    print("  Backtest CSV    → outputs/backtest_results.csv")

    return backtest_df


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("  WC2026 — MODEL TRAINING & BACKTESTING")
    print("=" * 58)

    wc        = load_features()
    feat_cols = get_features(wc)

    # ── A) Standard evaluation ────────────────────────────────
    result   = run_standard_evaluation(wc, feat_cols)
    lr       = result["lr"]
    xgb      = result["xgb"]
    poisson  = result["poisson"]
    metrics  = result["metrics"]

    # ── B) Backtest across all WC editions ───────────────────
    backtest_df = run_backtest(wc, feat_cols)

    # ── Save trained models (fit on ALL data including 2022) ─
    print("\n  Training final models on ALL data (incl. 2022) for 2026 prediction...")
    all_data = wc[wc["result_num"].notna()]
    lr_final  = BalancedLR()
    lr_final.fit(all_data[feat_cols], all_data["result_num"])

    xgb_final = WCXGBoost()
    xgb_final.fit(all_data[feat_cols], all_data["result_num"])

    pois_final = PoissonGoals()
    pois_final.fit(all_data[feat_cols], all_data["home_score"], all_data["away_score"])

    save_models(lr_final, xgb_final, pois_final, MODEL_DIR)

    # ── Final comparison table ────────────────────────────────
    print("\n" + "=" * 58)
    print("  FINAL COMPARISON (2022 test set)")
    print("=" * 58)
    cmp = pd.DataFrame(metrics)
    cmp["accuracy"] = cmp["accuracy"].round(4)
    print(cmp[["name", "accuracy", "log_loss", "brier"]].to_string(index=False))

    print("\n" + "=" * 58)
    print("  BACKTEST SUMMARY (leave-one-out, 1994–2022)")
    print("=" * 58)
    print(backtest_df[["year", "n_matches", "acc_lr", "acc_xgb", "acc_poisson"]].to_string(index=False))

    print("\n  All outputs saved to outputs/")
    print("  Models saved to models/")
    print("\n  Next step → run: python scripts/predict_2026.py")


if __name__ == "__main__":
    main()