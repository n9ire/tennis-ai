# 🎾 Tennis Match Winner Prediction

This project uses **XGBoost** to predict the winner of professional tennis matches using
pre-match data such as **Elo ratings, rankings, player attributes, and surface information**.

The model predicts whether **Player A wins the match** (`player_a_win = 1`).

---

## 📌 Problem Definition

Given two players in a tennis match, predict the probability that **Player A** wins.

This is a **binary classification** problem:
- `1` → Player A wins
- `0` → Player B wins

---

## 📂 Dataset Overview

Each row represents **one tennis match**, with players randomly assigned as **Player A** and **Player B**
to avoid positional bias.

### Target
player_a_win


### Features

#### Numeric Features
- `best_of`
- `minutes` *(remove for pre-match prediction)*
- `player_a_age`, `player_b_age`
- `player_a_rank`, `player_b_rank`
- `player_a_height`, `player_b_height`
- `player_a_elo`, `player_b_elo`
- `rank_diff`
- `age_diff`
- `height_diff`
- `elo_diff`

#### Categorical Features
- `surface` (Hard, Clay, Grass, Carpet)
- `round` (R32, QF, SF, F, etc.)
- `player_a_hand` (R/L)
- `player_b_hand` (R/L)

❌ Player names are excluded from training.

---

## 🧠 Model Choice: XGBoost

**Why XGBoost?**

- Handles non-linear interactions extremely well
- No need for feature scaling
- Excellent performance on tabular sports data
- Robust to noisy and correlated features

---

## ⚙️ Training Pipeline

The model is trained using an **sklearn Pipeline**:

1. **OneHotEncoding** for categorical variables
2. **Pass-through** numeric variables
3. **XGBoost Classifier** for prediction

Data is split **chronologically** (no shuffling) to prevent data leakage.

---

## 🏗️ Model Configuration

```python
xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42
)
```

## 📊 Model Performance

### Evaluation Metrics
The model is evaluated using the following metrics:

- **Accuracy** – proportion of correctly predicted match outcomes
- **ROC-AUC** – ability of the model to rank winning probabilities correctly  
  *(primary metric due to class imbalance and betting relevance)*

### Results

| Model                  | Accuracy | ROC-AUC |
|------------------------|----------|---------|
| Logistic Regression    | ~63%     | ~0.68   |
| Neural Network (MLP)   | ~57-64%  | ~0.62   |
| **XGBoost (Final)**    | **75–80%** | **0.82–0.87** |

> Performance varies slightly depending on season range, surface distribution, and feature availability.

---

## 🔍 Feature Importance

Top features by **gain importance** from XGBoost:

1. `elo_diff`
2. `player_a_elo`
3. `player_b_elo`
4. `rank_diff`
5. `surface`
6. `round`

Elo-based features dominate the prediction signal, confirming their effectiveness
in modeling player strength.

---

## 🚨 Notes & Best Practices

- Elo ratings must be calculated **strictly pre-match**
- Do **not shuffle** time-series sports data
- Remove `minutes` for true pre-match predictions
- Randomize Player A / Player B assignment to avoid positional bias
- Avoid player names as model features

---

## 🚀 Future Improvements

- Surface-specific Elo ratings
- Recent-form features (last N matches)
- Tournament-level weighting (Grand Slams vs ATP 250)
- Bayesian or Platt calibration for probability outputs
- Betting ROI simulation and bankroll modeling

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn xgboost
```

## 🏁 Conclusion

This project demonstrates that **XGBoost is a highly effective model for predicting tennis match outcomes**
when combined with well-engineered pre-match features such as **Elo ratings, ranking differences, surface type,
and player attributes**.

The final model consistently achieves **strong ROC-AUC performance (0.82–0.87)**, outperforming both
logistic regression and neural network baselines while remaining efficient and interpretable.

Key takeaways:
- Elo-based features provide the strongest predictive signal
- Tree-based models excel at capturing non-linear interactions in tennis data
- Proper time-aware data splitting is critical to avoid information leakage

---

## 🔬 Limitations

- The model does not account for in-match dynamics or injuries
- Elo ratings assume consistent player form between matches
- Performance may vary for early-round matches with sparse data
- Betting odds and market information are not included

---

## 📎 Reproducibility

To reproduce results:

1. Ensure Python **64-bit** is installed
2. Install dependencies
3. Run preprocessing to compute Elo ratings
4. Train the XGBoost model using chronological splits
5. Evaluate using ROC-AUC and accuracy metrics

Random seeds are fixed where applicable to ensure consistent results.

---

*Built for robust, data-driven tennis analytics.*


