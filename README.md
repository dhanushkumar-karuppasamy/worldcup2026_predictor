# FIFA World Cup 2026 — ML Prediction System

Predicts match outcomes and tournament winner using Elo ratings, rolling team form, and xG data.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Add a `.env` file in the project root:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

## Run in order

```bash
# 1. Download all data + build feature files
python scripts/build_pipeline.py

# 2. Train and evaluate models (2018 validation, 2022 test)
python scripts/train_models.py

# 3. Simulate the 2026 World Cup 10,000 times
python scripts/predict_2026.py
```

## Project structure

```
src/
  data_loader.py    ← downloads and cleans all 4 datasets
  features.py       ← Elo, rolling form, xG, feature matrix
  wc2026_config.py  ← 48 teams, groups, fixture list
  models.py         ← LR, XGBoost, Poisson model wrappers
  simulator.py      ← Monte Carlo tournament simulator (Phase 4)
  evaluate.py       ← metrics, calibration, reports

scripts/
  build_pipeline.py ← run once to build data/processed/
  train_models.py   ← trains and saves models
  predict_2026.py   ← 2026 predictions

data/
  raw/              ← downloaded from Kaggle (gitignored)
  processed/        ← Parquet files built by pipeline (gitignored)
```

## Model accuracy (2022 test set)

| Model             | Accuracy | Log Loss |
|-------------------|----------|----------|
| Logistic Regression | 39%   | 1.19     |
| XGBoost           | 45%      | 1.15     |
| Poisson Goals     | 50%      | —        |

These are realistic for international football prediction (3-class problem).
Academic benchmarks: 50–55% accuracy on World Cup data.

## Roadmap

- [x] Phase 1: Environment + project structure
- [x] Phase 2: Data pipeline (data_loader + features)
- [ ] Phase 3: Model training + backtesting
- [ ] Phase 4: Monte Carlo tournament simulator
- [ ] Phase 5: 2026 predictions + live updates