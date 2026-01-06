import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("merged_tennis_files/merged_file.csv")

# =========================
# 2. REMOVE NaNs (STRICT)
# =========================

required_cols = [
    'winner_name', 'loser_name',
    'winner_age', 'loser_age',
    'winner_rank', 'loser_rank',
    'winner_ht', 'loser_ht',
    'winner_hand', 'loser_hand',
    'surface', 'minutes', 'best_of', 'round',
    'tourney_date', 'score'
]

df = df.dropna(subset=required_cols).reset_index(drop=True)

# =========================
# 3. BASIC CLEANING
# =========================

df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
df['surface'] = df['surface'].str.upper()

# Remove walkovers
df = df[df['score'].notna()].reset_index(drop=True)

# Sort chronologically (IMPORTANT for Elo)
df = df.sort_values('tourney_date').reset_index(drop=True)

# =========================
# 4. RANDOMIZE PLAYER A / B
# =========================

np.random.seed(42)
mask = np.random.rand(len(df)) > 0.5

# =========================
# 5. BUILD ML DATAFRAME
# =========================

ml_df = pd.DataFrame()

# Match context
ml_df['surface'] = df['surface']
ml_df['best_of'] = df['best_of']
ml_df['round'] = df['round']
ml_df['minutes'] = df['minutes']

# Player A
ml_df['player_a_name'] = np.where(mask, df['loser_name'], df['winner_name'])
ml_df['player_a_age'] = np.where(mask, df['loser_age'], df['winner_age'])
ml_df['player_a_rank'] = np.where(mask, df['loser_rank'], df['winner_rank'])
ml_df['player_a_height'] = np.where(mask, df['loser_ht'], df['winner_ht'])
ml_df['player_a_hand'] = np.where(mask, df['loser_hand'], df['winner_hand'])

# Player B
ml_df['player_b_name'] = np.where(mask, df['winner_name'], df['loser_name'])
ml_df['player_b_age'] = np.where(mask, df['winner_age'], df['loser_age'])
ml_df['player_b_rank'] = np.where(mask, df['winner_rank'], df['loser_rank'])
ml_df['player_b_height'] = np.where(mask, df['winner_ht'], df['loser_ht'])
ml_df['player_b_hand'] = np.where(mask, df['winner_hand'], df['loser_hand'])

# Target
ml_df['player_a_win'] = np.where(mask, 0, 1)

# =========================
# 6. DIFFERENCE FEATURES
# =========================

ml_df['rank_diff'] = ml_df['player_a_rank'] - ml_df['player_b_rank']
ml_df['age_diff'] = ml_df['player_a_age'] - ml_df['player_b_age']
ml_df['height_diff'] = ml_df['player_a_height'] - ml_df['player_b_height']

# =========================
# 7. ELO FUNCTIONS
# =========================

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def update_elo(r, expected, score, k=32):
    return r + k * (score - expected)

# =========================
# 8. COMPUTE PRE-MATCH ELO
# =========================

elo = {}
START_ELO = 1500

player_a_elo = []
player_b_elo = []

for _, row in ml_df.iterrows():
    a = row['player_a_name']
    b = row['player_b_name']

    elo_a = elo.get(a, START_ELO)
    elo_b = elo.get(b, START_ELO)

    # Store PRE-match Elo
    player_a_elo.append(elo_a)
    player_b_elo.append(elo_b)

    # Match outcome
    score_a = row['player_a_win']
    score_b = 1 - score_a

    exp_a = expected_score(elo_a, elo_b)
    exp_b = expected_score(elo_b, elo_a)

    # Update Elo AFTER match
    elo[a] = update_elo(elo_a, exp_a, score_a)
    elo[b] = update_elo(elo_b, exp_b, score_b)

# Add Elo features
ml_df['player_a_elo'] = player_a_elo
ml_df['player_b_elo'] = player_b_elo
ml_df['elo_diff'] = ml_df['player_a_elo'] - ml_df['player_b_elo']

# =========================
# 9. FINAL CLEANUP
# =========================

ml_df = ml_df.dropna().reset_index(drop=True)

# =========================
# 10. SAVE FINAL DATASET
# =========================

ml_df.to_csv("tennis_matches_ml_with_elo.csv", index=False)

print("✅ ML dataset with Elo ratings created successfully!")
