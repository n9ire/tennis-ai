import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("merged_tennis_files/merged_file.csv")

# =========================
# 2. DROP NaNs (STRICT)
# =========================

required_cols = [
    'winner_name', 'loser_name',
    'winner_age', 'loser_age',
    'winner_rank', 'loser_rank',
    'winner_ht', 'loser_ht',
    'winner_hand', 'loser_hand',
    'surface', 'best_of', 'round',
    'tourney_date', 'score', 'minutes'
]

df = df.dropna(subset=required_cols).reset_index(drop=True)

# =========================
# 3. BASIC CLEANING
# =========================

df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
df['surface'] = df['surface'].str.upper()
df['round'] = df['round'].str.upper()

# Remove walkovers / retirements
df = df[~df['score'].str.contains('W/O', regex=False)]
df = df[~df['score'].str.contains('RET', regex=False)]

df = df.sort_values('tourney_date').reset_index(drop=True)

# =========================
# 4. RANDOMIZE PLAYER A / B
# =========================

np.random.seed(42)
swap = np.random.rand(len(df)) > 0.5

# =========================
# 5. BUILD ML DATAFRAME
# =========================

ml = pd.DataFrame()

# Match context
ml['tourney_date'] = df['tourney_date']
ml['surface'] = df['surface']
ml['round'] = df['round']
ml['best_of'] = df['best_of']
ml['minutes'] = df['minutes']

# Player A
ml['player_a_name'] = np.where(swap, df['loser_name'], df['winner_name'])
ml['player_a_age'] = np.where(swap, df['loser_age'], df['winner_age'])
ml['player_a_rank'] = np.where(swap, df['loser_rank'], df['winner_rank'])
ml['player_a_height'] = np.where(swap, df['loser_ht'], df['winner_ht'])
ml['player_a_hand'] = np.where(swap, df['loser_hand'], df['winner_hand'])

# Player B
ml['player_b_name'] = np.where(swap, df['winner_name'], df['loser_name'])
ml['player_b_age'] = np.where(swap, df['winner_age'], df['loser_age'])
ml['player_b_rank'] = np.where(swap, df['winner_rank'], df['loser_rank'])
ml['player_b_height'] = np.where(swap, df['winner_ht'], df['loser_ht'])
ml['player_b_hand'] = np.where(swap, df['winner_hand'], df['loser_hand'])

# Target
ml['player_a_win'] = np.where(swap, 0, 1)

# =========================
# 6. DIFFERENCE FEATURES
# =========================

ml['rank_diff'] = ml['player_a_rank'] - ml['player_b_rank']
ml['age_diff'] = ml['player_a_age'] - ml['player_b_age']
ml['height_diff'] = ml['player_a_height'] - ml['player_b_height']

# =========================
# 7. FINAL CLEANUP
# =========================

ml = ml.dropna().reset_index(drop=True)

# =========================
# 8. SAVE DATASET
# =========================

ml.to_csv("merged_tennis_files/tennis_ml_player_ab.csv", index=False)

print("✅ Player A / Player B dataset created successfully!")
