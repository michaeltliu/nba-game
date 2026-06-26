import pandas as pd

players_df = pd.read_csv("/Users/michaeltliu/.cache/kagglehub/datasets/eoinamoore/historical-nba-data-and-player-box-scores/versions/515/Players.csv")
stats_df = pd.read_csv("/Users/michaeltliu/.cache/kagglehub/datasets/eoinamoore/historical-nba-data-and-player-box-scores/versions/515/PlayerStatistics.csv")
stats_df['date'] = pd.to_datetime(stats_df['gameDate'])
stats26 = stats_df[(stats_df['date'].between('2025-10-20', '2026-04-18')) & (stats_df['gameType'].isin(['Regular Season']))].copy()

# Convert numMinutes to numeric, coercing any non-numeric values to NaN
stats26['numMinutes_num'] = pd.to_numeric(stats26['numMinutes'], errors='coerce')

# Filter for games actually logged/played (where minutes played is greater than 0)
stats26_played = stats26[stats26['numMinutes_num'] > 0]

# List of stat columns to aggregate
stat_cols = [
    'points', 'assists', 'blocks', 'steals', 'reboundsTotal', 'turnovers',
    'fieldGoalsAttempted', 'fieldGoalsMade', 'threePointersAttempted', 'threePointersMade',
    'freeThrowsAttempted', 'freeThrowsMade'
]

# Calculate every player's average stats this season (sum of stats divided by games played)
# We also include the count of gameId to represent the number of games played
player_averages = stats26_played.groupby('personId').agg({
    'firstName': 'first',
    'lastName': 'first',
    'gameId': 'count',
    **{col: 'mean' for col in stat_cols}
}).reset_index().rename(columns={'gameId': 'gamesPlayed'})

# Cast personId to integer for clean formatting
player_averages['personId'] = player_averages['personId'].astype(int)

# Merge player positions (guard, forward, center) from players_df
players_positions = players_df[['personId', 'guard', 'forward', 'center']]
player_averages = player_averages.merge(players_positions, on='personId', how='left')

# Fill any missing position values with 0 and convert them to integer
player_averages[['guard', 'forward', 'center']] = player_averages[['guard', 'forward', 'center']].fillna(0).astype(int)

# Calculate True Shooting percentage (TS%)
denom = 2 * (player_averages['fieldGoalsAttempted'] + 0.44 * player_averages['freeThrowsAttempted'])
player_averages['ts'] = player_averages['points'] / denom
player_averages['ts'] = player_averages['ts'].fillna(0.0)

# Calculate Fantasy Points
player_averages['fantasyPoints'] = (
    player_averages['points'] +
    player_averages['reboundsTotal'] +
    1.5 * player_averages['assists'] +
    4 * player_averages['steals'] +
    4 * player_averages['blocks'] -
    2 * player_averages['turnovers']
)

# Filter out players with no valid position
player_averages = player_averages[player_averages[['guard', 'forward', 'center']].any(axis=1)]

# Sort players by fantasyPoints descending
player_averages = player_averages.sort_values(by='fantasyPoints', ascending=False)

# Rearrange columns to place positions right after lastName
cols = list(player_averages.columns)
last_name_idx = cols.index('lastName')
remaining_cols = [c for c in cols if c not in ['guard', 'forward', 'center']]
new_col_order = remaining_cols[:last_name_idx+1] + ['guard', 'forward', 'center'] + remaining_cols[last_name_idx+1:]
player_averages = player_averages[new_col_order]

# Save the resulting dataframe to a CSV in the same directory
output_csv_path = 'player_averages_2025_26.csv'
player_averages.to_csv(output_csv_path, index=False)
print(f"Successfully calculated averages for {len(player_averages)} players and saved to {output_csv_path}")