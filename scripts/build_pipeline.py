# scripts/build_pipeline.py
# ─────────────────────────────────────────────────────────────
# Run this ONCE to download all data, build all features,
# and save clean Parquet files to data/processed/.
#
# Usage (from project root, with venv activated):
#   python scripts/build_pipeline.py
#
# After this runs successfully, all other scripts load from
# data/processed/ — fast, no re-downloading needed.
# ─────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data_loader import load_all_data, PROC_DIR
from src.features import (
    build_elo,
    build_rolling_form,
    build_xg_features,
    build_prev_wc_features,
    build_feature_matrix,
    get_features,
)


def main():
    print("=" * 60)
    print("  WC2026 PREDICTION PIPELINE — BUILD STAGE")
    print("=" * 60)

    # ── Step 1: Load raw data ────────────────────────────────
    data = load_all_data()
    results   = data["results"]
    wcm       = data["wcm"]
    wcs       = data["wcs"]
    standings = data["standings"]
    xg        = data["xg"]

    # ── Step 2: Build Elo ────────────────────────────────────
    results_with_elo, final_elo_map = build_elo(results)

    # Save current Elo ratings (useful for 2026 predictions)
    elo_df = pd.DataFrame([
        {"team": k, "elo": v} for k, v in sorted(final_elo_map.items())
    ])
    elo_df.to_parquet(PROC_DIR / "elo_ratings.parquet", index=False)
    print(f"  Saved elo_ratings.parquet ({len(elo_df)} teams)")

    # ── Step 3: Rolling form ─────────────────────────────────
    long_feat = build_rolling_form(results_with_elo)
    long_feat.to_parquet(PROC_DIR / "rolling_form.parquet", index=False)
    print(f"  Saved rolling_form.parquet ({len(long_feat)} rows)")

    # ── Step 4: xG features ──────────────────────────────────
    xg_feat = build_xg_features(xg)
    if not xg_feat.empty:
        xg_feat.to_parquet(PROC_DIR / "xg_features.parquet", index=False)
        print(f"  Saved xg_features.parquet ({len(xg_feat)} rows)")
    else:
        print("  xG features empty — skipping.")

    # ── Step 5: Previous WC features ─────────────────────────
    prev_wc = build_prev_wc_features(standings)
    if not prev_wc.empty:
        prev_wc.to_parquet(PROC_DIR / "prev_wc.parquet", index=False)
        print(f"  Saved prev_wc.parquet ({len(prev_wc)} rows)")

    # ── Step 6: Full feature matrix ──────────────────────────
    wc_features = build_feature_matrix(
        results_with_elo, long_feat,
        xg_feat if not xg_feat.empty else pd.DataFrame(),
        prev_wc if not prev_wc.empty else pd.DataFrame(),
        wcm, wcs
    )
    wc_features.to_parquet(PROC_DIR / "wc_features.parquet", index=False)
    print(f"  Saved wc_features.parquet ({len(wc_features)} rows)")

    # ── Step 7: Train/Val/Test split summary ─────────────────
    feat_cols = get_features(wc_features)
    train = wc_features[wc_features["year"] < 2018]
    val   = wc_features[wc_features["year"] == 2018]
    test  = wc_features[wc_features["year"] == 2022]

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — Summary")
    print("=" * 60)
    print(f"  Total WC matches in matrix : {len(wc_features)}")
    print(f"  Train set (pre-2018)        : {len(train)} matches")
    print(f"  Validation set (2018)       : {len(val)} matches")
    print(f"  Test set (2022)             : {len(test)} matches")
    print(f"  Features ready for modelling: {len(feat_cols)}")
    print(f"\n  Years covered: {sorted(wc_features.year.unique())}")
    print("\n  Next step → run: python scripts/train_models.py")
    print("=" * 60)


if __name__ == "__main__":
    main()