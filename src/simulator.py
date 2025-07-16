import joblib
import numpy as np

# Load model and encoder
model = joblib.load("src/model.pkl")
le_team = joblib.load("src/label_encoder.pkl")

TEAM_FLAGS = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "Germany": "🇩🇪",
    "England": "🏴", "Portugal": "🇵🇹", "Spain": "🇪🇸", "Italy": "🇮🇹",
    "Netherlands": "🇳🇱", "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾",
    "USA": "🇺🇸", "Mexico": "🇲🇽", "Japan": "🇯🇵", "South Korea": "🇰🇷",
    "Australia": "🇦🇺", "Canada": "🇨🇦", "Qatar": "🇶🇦", "Senegal": "🇸🇳",
    "Morocco": "🇲🇦", "Cameroon": "🇨🇲", "Ghana": "🇬🇭", "Ivory Coast": "🇨🇮",
    "Saudi Arabia": "🇸🇦", "Iran": "🇮🇷", "New Zealand": "🇳🇿", "Ecuador": "🇪🇨",
    "Peru": "🇵🇪", "Chile": "🇨🇱", "Switzerland": "🇨🇭", "Poland": "🇵🇱",
    "Denmark": "🇩🇰", "Sweden": "🇸🇪", "Norway": "🇳🇴", "Ukraine": "🇺🇦",
    "Turkey": "🇹🇷", "Tunisia": "🇹🇳", "South Africa": "🇿🇦", "Nigeria": "🇳🇬",
    "Egypt": "🇪🇬", "Jamaica": "🇯🇲", "Panama": "🇵🇦", "Paraguay": "🇵🇾",
    "Colombia": "🇨🇴", "Honduras": "🇭🇳", "Algeria": "🇩🇿", "Czech Republic": "🇨🇿"
}

def get_flag(team):
    return TEAM_FLAGS.get(team, "🏳️")  # fallback if missing


# Simulate a match and return winner + predicted probability
def simulate_match(team1, team2):
    try:
        t1 = le_team.transform([team1])[0]
        t2 = le_team.transform([team2])[0]
    except ValueError as e:
        raise ValueError(f"Unknown team: {e}")

    probs = model.predict_proba([[t1, t2]])[0]
    winner = team1 if probs[0] > probs[1] else team2
    return {
       "match": f"{get_flag(team1)} {team1} vs {get_flag(team2)} {team2}",
        "winner": winner,
        "prob": max(probs[0], probs[1])
    }

# Simulate a round of matches
def simulate_round(matchups):
    return [simulate_match(t1, t2) for t1, t2 in matchups]

# Build the next round from winners
def build_next_round(winners):
    if len(winners) % 2 != 0:
        print("⚠️ Warning: Odd number of winners. One team is left out.")
        winners = winners[:-1]  # Drop last team

    return [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]


# Full tournament simulation
def simulate_tournament(initial_round, verbose=False):
    matches = initial_round
    all_rounds = []

    while len(matches) > 0:
        round_data = simulate_round(matches)
        all_rounds.append(round_data)

        winners = [match["winner"] for match in round_data]
        matches = build_next_round(winners)

        if len(winners) == 1:
            break

    final_winner = winners[0]
    if verbose:
        print(f"\n🏆 Champion: {final_winner}")
    return final_winner, all_rounds

# Manual test run
if __name__ == "__main__":
    demo_round = [
        ("Brazil", "Germany"),
        ("Argentina", "France"),
        ("England", "Netherlands"),
        ("Spain", "Croatia"),
        ("Portugal", "USA"),
        ("Italy", "Switzerland"),
        ("Japan", "South Korea"),
        ("Uruguay", "Mexico"),
        ("Morocco", "Senegal"),
        ("Poland", "Cameroon"),
        ("Denmark", "Ghana"),
        ("Australia", "Canada"),
        ("South Korea", "Ecuador"),
        ("Qatar", "Peru"),
        ("Belgium", "Saudi Arabia"),
        ("Mexico", "New Zealand"),
    ]
    simulate_tournament(demo_round, verbose=True)


