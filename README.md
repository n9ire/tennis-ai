# tennis-ai

This README is a concise guide describing the models included in this repo, how they work, and how to run them.

**Overview**
- **Purpose**: Predict the winner of an ATP tennis match (binary: Player A wins or loses) using pre-match features.
- **Main models**: XGBoost classifier (tabular, tree-based) and simple neural-network experiments (in `nns/`).

**Data**
- **Source files**: per-year match CSVs are in [tennis](tennis/).
- **Processed datasets**: merged and feature-engineered CSVs are in [merged_tennis_files](merged_tennis_files/).
- **Key features**: pre-match Elo ratings, ranks, ages, heights, surface, round, and engineered diffs (e.g., `elo_diff`).

**How the models work**
- **Elo preprocessing**: `scripts/elo.py` computes player Elo ratings strictly using matches prior to each match (no leakage).
- **Feature engineering**: create numeric diffs (`elo_diff`, `rank_diff`, etc.), keep pre-match-only columns, one-hot encode categorical fields (`surface`, `round`).
- **XGBoost pipeline**: numeric features pass through; categorical features are one-hot encoded; final estimator is `xgboost.XGBClassifier` trained on chronological splits (no shuffling) to avoid leakage.
- **Neural nets**: notebooks under `nns/` contain small MLP experiments — useful as baselines but typically underperform tuned XGBoost on this tabular task.

**Why XGBoost?**
- Handles non-linear interactions and missing values well.
- No feature scaling required and strong performance on tabular sports data.

**Quick run instructions**
- Create a virtual environment and install requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- If you don't have `requirements.txt`, install minimal deps:

```bash
pip install pandas numpy scikit-learn xgboost
```

- Example: run the XGBoost training script (uses processed CSVs from `merged_tennis_files`):

```bash
python scripts/xgboost.py --data merged_tennis_files/tennis_matches_ml_with_elo.csv --out models/xgb.pkl
```

- Compute Elo ratings (if you need to recreate features):

```bash
python scripts/elo.py --input tennis/ --output merged_tennis_files/tennis_matches_with_elo.csv
```

**Where to look**
- `scripts/xgboost.py` — example training entrypoint and hyperparameters.
- `scripts/elo.py` — Elo calculation utilities (pre-match Elo construction).
- `nns/` — neural-net experiment notebooks.
- `xgboost-models/` — experimentation notebooks and saved artifacts.

**Notes & best practices**
- Always compute Elo and features using only data available before each match.
- Use chronological train/validation/test splits to avoid leakage.
- Remove in-match columns (e.g., `minutes`) for pre-match models.

**Want changes?**
- I kept this README focused on "how it works" and how to run training/inference. Tell me if you want a shorter summary, example commands added to scripts, or a `requirements.txt` generated.

---
Updated README to focus on models, data, and run instructions.