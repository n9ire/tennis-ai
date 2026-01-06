# tennis-ai

This README is a concise guide describing the models included in this repo, how they work, and how to run them.

## 📖 Dictionary
- [Overview](#Overview)
- [Performance](#Performance)
- [How the Models Work](#How-the-models-work)
- [Model Selection](#Model-Selection)
- [Elo Rating System](#Elo-Rating-System)
- [Links](#Links)
- [Future Features](#SOON-TO-COME)

## Overview
- **Purpose**: Predict the winner of an ATP tennis match (binary: Player A wins or loses) using pre-match features.
- **Main models**: XGBoost classifier (tabular, tree-based) and simple neural-network experiments (in `nns/`).

## Performance
| Model | Accuracy | ROC AUC |
|------|--------|----------------|
| XGBoost | 70-75% | 80-85% |
| Neural Network (PyTorch) | 57-64%| 70% |

## How the models work
- **Elo preprocessing**: `scripts/elo.py` computes player Elo ratings strictly using matches prior to each match (no leakage).
- **Feature engineering**: create numeric diffs (`elo_diff`, `rank_diff`, etc.), keep pre-match-only columns, one-hot encode categorical fields (`surface`, `round`).
- **XGBoost pipeline**: numeric features pass through; categorical features are one-hot encoded; final estimator is `xgboost.XGBClassifier` trained on chronological splits (no shuffling) to avoid leakage.
- **Neural nets**: notebooks under `nns/` contain small MLP experiments — useful as baselines but typically underperform tuned XGBoost on this tabular task.

## Model Selection

Predicting tennis match outcomes is challenging due to non-linear relationships, player variability, surface effects, and time-based trends. Both **XGBoost** and **Neural Networks** are well-suited for this task, each offering distinct advantages.

---

### Why XGBoost Works Well

**Structured Data Performance**  
Tennis datasets are typically tabular, including rankings, surface records, head-to-head stats, and recent form. XGBoost is highly optimized for this type of structured data.

**Non-Linear Feature Interactions**  
XGBoost naturally captures interactions such as surface strengths, ranking gaps adjusted by form, and fatigue effects without extensive manual feature engineering.

**Robustness on Limited or Noisy Data**  
Sports data often contains missing or noisy values. XGBoost handles these cases well and performs strongly on small to medium-sized datasets.

**Interpretability**  
Feature importance and decision paths help explain predictions, making the model easier to validate and analyze.

---

### Why Neural Networks Work Well

**Complex Pattern Learning**  
Neural Networks model deep, non-linear relationships such as momentum, latent player form, and indirect performance signals.

**Temporal Awareness**  
When match data is ordered chronologically, Neural Networks can capture trends, streaks, and form progression that influence outcomes.

**Automatic Feature Learning**  
Neural Networks reduce reliance on handcrafted features by learning internal representations of players and match context.

**Scalability**  
They scale effectively with larger datasets and additional inputs, such as expanded match statistics or external signals.

---

### Complementary Strengths

| Aspect | XGBoost | Neural Network |
|------|--------|----------------|
| Best for | Tabular, structured data | Complex & temporal patterns |
| Data size | Small to medium | Medium to large |
| Interpretability | High | Low |
| Training speed | Fast | Slower |
| Feature engineering | Important | Less required |

Rather than competing, these models are often complementary and can be combined in ensemble approaches for improved performance.


## Elo Rating System

**Definition**
- The Elo rating system is used to measure a player’s relative skill level by updating their rating after each match based on the opponent’s strength and the match result. It provides a dynamic way to estimate how likely a player is to win against another.

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
- 

---

## SOON TO COME
- An online instance of this model (**cough cough** https://tennis.noire.li/)





