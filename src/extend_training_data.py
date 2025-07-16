import pandas as pd
import random

# 48 expected 2026 teams
teams_2026 = [
    "Argentina", "Brazil", "France", "Germany", "England", "Portugal", "Spain", "Italy",
    "Netherlands", "Belgium", "Croatia", "Uruguay", "USA", "Mexico", "Japan", "South Korea",
    "Australia", "Canada", "Qatar", "Senegal", "Morocco", "Cameroon", "Ghana", "Ivory Coast",
    "Saudi Arabia", "Iran", "New Zealand", "Ecuador", "Peru", "Chile", "Switzerland", "Poland",
    "Denmark", "Sweden", "Norway", "Ukraine", "Turkey", "Tunisia", "South Africa", "Nigeria",
    "Egypt", "Jamaica", "Panama", "Paraguay", "Colombia", "Honduras", "Algeria", "Czech Republic"
]

# Load your original match dataset
data = pd.read_csv("data/processed/cleaned_matches.csv")

# Get list of all teams already in training data
teams_in_data = set(data["Home Team Name"]).union(set(data["Away Team Name"]))

# Find missing teams
missing = [team for team in teams_2026 if team not in teams_in_data]
print("Missing teams:", missing)

# Generate 2 fake matches per missing team vs known teams
known_teams = list(teams_in_data)
fake_rows = []

for team in missing:
    for _ in range(2):
        opponent = random.choice(known_teams)
        fake_rows.append({
            "Year": 2026,
            "Stage": "Friendly",
            "Home Team Name": team,
            "Away Team Name": opponent,
            "Home Team Goals": random.randint(0, 2),
            "Away Team Goals": random.randint(1, 3),
            "Result": random.choice(["HomeWin", "AwayWin", "Draw"])
        })

# Add to the existing dataset
df_fake = pd.DataFrame(fake_rows)
df_extended = pd.concat([data, df_fake], ignore_index=True)

# Save new extended dataset
df_extended.to_csv("data/processed/cleaned_matches_extended.csv", index=False)
print("Extended data saved to data/processed/cleaned_matches_extended.csv")
