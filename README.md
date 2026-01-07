<h1 align="center">tennis-ai</h1>

This README is a straightforward guide to what’s in this repo, how the models work, and how to run or extend them.

**Origin:**
- This project started after I watched a video by [Green Code](https://www.youtube.com/watch?v=N4JDlSTMOck) where he built a Decision Tree model and later an XGBoost model that reached around 80% accuracy. I wanted to see how far I could realistically push similar ideas, and whether I could get anywhere near (or even past) the kind of models Wimbledon and IBM probably pay absurd amounts of money for.

## 📖 Dictionary
- [Overview](#Overview)
- [Performance](#Performance)
- [How the Models Work](#How-the-models-work)
- [Model Selection](#Model-Selection)
- [Elo Rating System](#Elo-Rating-System)
- [Links](#Links)
- [Future Features](#SOON-TO-COME)

## Overview
- **Purpose**: Predict the winner of an ATP tennis match using binary classification based entirely on pre-match features.
- **Main models**: An XGBoost classifier (tree-based, tabular) and some experimental neural networks located in `nns/`.

---

## Performance
| Model | Accuracy | ROC AUC |
|------|--------|----------------|
| XGBoost | 70-75% | 80-85% |
| Neural Network (PyTorch) | 57-64%| 70% |

**ROC AUC (Area Under the Receiver Operating Characteristic Curve)** is a common metric for evaluating binary classifiers. It measures how well a model separates the positive and negative classes across all possible decision thresholds. An AUC of 1.0 is perfect, 0.5 is no better than random guessing, and higher values mean better discrimination. In simple terms, it tells you how good the model is at ranking winners above losers.

---

## How the models work

### Elo preprocessing (core signal)
`scripts/elo.py` computes **pre-match Elo ratings** by processing matches in **strict chronological order**, which is critical to avoid any form of data leakage.

For each match:
- Each player’s **global Elo** is read *before* the match starts
- A **surface-specific Elo** (hard / clay / grass) is also tracked
- Elo ratings are updated **only after** the match result is known

This produces four Elo-related features:
- `player_a_elo`, `player_b_elo`
- `elo_diff`
- `elo_surface_diff`

Elo ratings capture underlying player strength in a way that raw rankings, age, or height simply can’t.  
In practice, adding Elo features improves out-of-sample accuracy by roughly **4–7 percentage points** compared to training without Elo.

---

### Feature engineering
All features are built using **only information available before the match**.

**Numeric difference features**
- `elo_diff`
- `elo_surface_diff`
- `rank_diff`
- `age_diff`
- `height_diff`

Using difference-based features helps reduce scale issues and encourages the model to learn *relative* advantages instead of absolute values.

**Categorical features**
- `surface`
- `round`
- `player_a_hand`
- `player_b_hand`

Categorical features are **one-hot encoded** during training.

Player names and match dates are kept in the dataset for **Elo calculation, joins, and analysis**, but they are **explicitly excluded** from model training.

---

### XGBoost pipeline
The main model is a gradient-boosted decision tree using `xgboost.XGBClassifier`.

Pipeline overview:
- Numeric features are passed through as-is
- Categorical features are one-hot encoded
- The model is trained using **chronological train/test splits** (no random shuffling)

Chronological splitting ensures the model is always tested on **future matches**, which prevents time-based leakage that would otherwise inflate results.

With Elo and engineered features included, the XGBoost model consistently reaches:
- **~70–75% accuracy**
- **~0.75–0.80 ROC AUC**

For tennis match prediction, which is inherently noisy and unpredictable, this is a solid result.

---

### Neural network baselines
Exploratory neural network models live in notebooks under `nns/`.

These are mostly small multi-layer perceptrons (MLPs) trained on the same engineered features. They’re useful as baselines, but they generally **underperform well-tuned XGBoost models** on this dataset due to:
- Relatively limited data per player
- Weaker inductive bias for tabular feature interactions
- Higher sensitivity to noise and feature scaling

---

## Spiders Web

- [My XGBoost Models](https://github.com/n9ire/tennis-ai/tree/main/xgboost-models)
- [My Neural Network Models](https://github.com/n9ire/tennis-ai/tree/main/nns)
- [My Custom Merged Tennis Datasets](https://github.com/n9ire/tennis-ai/tree/main/merged_tennis_files)
- [Visual XGBoost Trees](https://github.com/n9ire/tennis-ai/tree/main/xgboost-models/xgboost-visuals)
- [Plotting Images](https://github.com/n9ire/tennis-ai/tree/main/images)

---

### Why XGBoost Works Well

**Structured Data Performance**  
Tennis data is mostly tabular: rankings, surfaces, form indicators, and head-to-head stats. XGBoost is built for exactly this kind of data.

**Non-Linear Feature Interactions**  
XGBoost naturally learns interactions like surface-specific strength, ranking gaps adjusted by form, and subtle matchup effects without a ton of manual work.

**Robustness on Limited or Noisy Data**  
Sports data is messy. XGBoost handles missing values and noise well and performs reliably on small to medium datasets.

**Interpretability**  
Feature importance scores and tree structures make it easier to understand *why* the model is making certain predictions.

---

### Why Neural Networks Work Well

**Complex Pattern Learning**  
Neural networks can model deep, non-linear relationships such as momentum, latent form, and indirect performance signals.

**Temporal Awareness**  
When data is ordered chronologically, neural networks can learn trends, streaks, and form changes over time.

**Automatic Feature Learning**  
They can reduce reliance on handcrafted features by learning internal representations of players and match context.

**Scalability**  
With more data and richer inputs, neural networks can scale very effectively.

---

### Each Models Strengths

| Aspect | XGBoost | Neural Network |
|------|--------|----------------|
| Best for | Tabular, structured data | Complex & temporal patterns |
| Data size | Small to medium | Medium to large |
| Interpretability | High | Low |
| Training speed | Fast | Slower |
| Feature engineering | Important | Less required |

Rather than competing directly, these approaches are often complementary. In practice, they can be combined in ensemble models to squeeze out even better performance.

---

## Elo Rating System

**Definition**
- The Elo rating system is used to measure a player’s relative skill level by updating their rating after each match based on the opponent’s strength and the match result. It provides a dynamic way to estimate how likely a player is to win against another.

**Why was this implemented with the models (more specifically the XGBoost model)**
- An Elo rating system improves an XGBoost model’s accuracy by encoding a player’s latent skill into a single, noise-resistant feature that reflects long-term performance better than raw rankings or stats. When computed pre-match and combined with surface-specific Elo, it gives XGBoost a strong, leakage-free signal that significantly improves generalization on future matches.

**How it work's**
- Elo estimates the expected outcome between two players based on their ratings:

```math
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
```

- Ratings are updated after a match using:

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

---

## Links
- [ATP Matches Datasets](https://github.com/JeffSackmann/tennis_atp)
- [XGBoost Python Module](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)
- [PyTorch](https://pytorch.org/)

---

## Plots

<div align="center">
  <img src="images/elo_dataset_pp_2026-01-06 18:38:32.635334.png" width="400" />
  <img src="images/elo_dataset_pp_2026-01-06 19:04:27.943331.png" width="400" />
</div>

---

## SOON TO COME

- An online instance of this model (**cough cough** https://tennis.noire.li/)


