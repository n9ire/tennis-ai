import pandas as pd
import numpy as np

# =========================
# 1. LOAD PLAYER A / B DATA
# =========================

ml = pd.read_csv("merged_tennis_files/tennis_ml_player_ab.csv")
ml['tourney_date'] = pd.to_datetime(ml['tourney_date'])

# Ensure chronological order
ml = ml.sort_values('tourney_date').reset_index(drop=True)

# =========================
# 2. ELO FUNCTIONS
# =========================

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def update_elo(r, expected, score, k=32):
    return r + k * (score - expected)

# =========================
# 3. INITIALIZE ELO STORAGE
# =========================

START_ELO = 1500

elo_global = {}             # {player: rating}
elo_surface = {}            # {(player, surface): rating}

a_elo, b_elo = [], []
a_elo_surf, b_elo_surf = [], []

# =========================
# 4. COMPUTE PRE-MATCH ELO
# =========================

for _, row in ml.iterrows():
    a = row['player_a_name']
    b = row['player_b_name']
    surface = row['surface']

    # Global Elo
    ea = elo_global.get(a, START_ELO)
    eb = elo_global.get(b, START_ELO)

    # Surface Elo
    esa = elo_surface.get((a, surface), START_ELO)
    esb = elo_surface.get((b, surface), START_ELO)

    # Store PRE-match values
    a_elo.append(ea)
    b_elo.append(eb)
    a_elo_surf.append(esa)
    b_elo_surf.append(esb)

    # Match outcome
    score_a = row['player_a_win']
    score_b = 1 - score_a

    # Expected scores
    exp_a = expected_score(ea, eb)
    exp_b = expected_score(eb, ea)

    exp_sa = expected_score(esa, esb)
    exp_sb = expected_score(esb, esa)

    # Update global Elo
    elo_global[a] = update_elo(ea, exp_a, score_a)
    elo_global[b] = update_elo(eb, exp_b, score_b)

    # Update surface Elo
    elo_surface[(a, surface)] = update_elo(esa, exp_sa, score_a)
    elo_surface[(b, surface)] = update_elo(esb, exp_sb, score_b)

# =========================
# 5. ADD ELO FEATURES
# =========================

ml['player_a_elo'] = a_elo
ml['player_b_elo'] = b_elo
ml['elo_diff'] = ml['player_a_elo'] - ml['player_b_elo']

ml['player_a_elo_surface'] = a_elo_surf
ml['player_b_elo_surface'] = b_elo_surf
ml['elo_surface_diff'] = (
    ml['player_a_elo_surface'] - ml['player_b_elo_surface']
)

# =========================
# 6. FINAL CLEANUP & SAVE
# =========================

ml.to_csv("merged_tennis_files/tennis_ml_player_ab_with_elo.csv", index=False)

print("🔥 Elo features added successfully!")
