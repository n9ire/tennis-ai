# tennis-ai

This README is a concise guide describing the models included in this repo, how they work, and how to run them.

## 📖 Dictionary
- [Overview](#Overview)
- [Data](#Data)
- [How the Models Work](#How-the-models-work)
- [XGBoost's Advantages](#XGBoost-Advantages)
- [Elo Rating System](#Elo-Rating-System)

### Overview
- **Purpose**: Predict the winner of an ATP tennis match (binary: Player A wins or loses) using pre-match features.
- **Main models**: XGBoost classifier (tabular, tree-based) and simple neural-network experiments (in `nns/`).

### Data
- **Source files**: per-year match CSVs are in [atp_matches](atp_matches/).
- **Processed datasets**: merged and feature-engineered CSVs are in [merged_tennis_files](merged_tennis_files/).
- **Key features**: pre-match Elo ratings, ranks, ages, heights, surface, round, and engineered diffs (e.g., `elo_diff`).

### How the models work
- **Elo preprocessing**: `scripts/elo.py` computes player Elo ratings strictly using matches prior to each match (no leakage).
- **Feature engineering**: create numeric diffs (`elo_diff`, `rank_diff`, etc.), keep pre-match-only columns, one-hot encode categorical fields (`surface`, `round`).
- **XGBoost pipeline**: numeric features pass through; categorical features are one-hot encoded; final estimator is `xgboost.XGBClassifier` trained on chronological splits (no shuffling) to avoid leakage.
- **Neural nets**: notebooks under `nns/` contain small MLP experiments — useful as baselines but typically underperform tuned XGBoost on this tabular task.

### XGBoost Advantages
- Handles non-linear interactions and missing values well.
- No feature scaling required and strong performance on tabular sports data.

### Elo Rating System

Elo estimates the expected outcome between two players based on their ratings:

```math
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
```

Ratings are updated after a match using:

```math
R_A' = R_A + K(S_A - E_A)
```

**Symbol Definitions:**
- Ra — current rating of player A  
- Rb — current rating of player B  
- Ea — expected score for player A  
- Sa — actual score for player A  
- Ra' — updated rating of player A  
- K — rating update factor  


### SOON TO COME
- An online instance of this model (**cough cough** https://tennis.noire.li/)




