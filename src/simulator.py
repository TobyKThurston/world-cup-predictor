import joblib
import numpy as np
import random

# Load model and encoder
model = joblib.load("src/model.pkl")
le_team = joblib.load("src/label_encoder.pkl")

# Basic 16-team knockout bracket
initial_round = [
    ("Brazil", "Germany"),
    ("Argentina", "France"),
    ("England", "Netherlands"),
    ("Spain", "Croatia"),
    ("Portugal", "USA"),
    ("Italy", "Switzerland"),
    ("Japan", "South Korea"),
    ("Uruguay", "Mexico"),
]

def simulate_match(team1, team2):
    try:
        t1 = le_team.transform([team1])[0]
        t2 = le_team.transform([team2])[0]
    except ValueError as e:
        print(f"Unknown team: {e}")
        return random.choice([team1, team2])  # fallback

    probs = model.predict_proba([[t1, t2]])[0]
    classes = model.classes_

    prob_dict = dict(zip(classes, probs))

    # Simulate based on probabilities
    rand = random.random()
    if rand < prob_dict["HomeWin"]:
        return team1
    elif rand < prob_dict["HomeWin"] + prob_dict["AwayWin"]:
        return team2
    else:
        return random.choice([team1, team2])  # for draws, random winner

def simulate_round(matches):
    winners = []
    for team1, team2 in matches:
        winner = simulate_match(team1, team2)
        print(f"{team1} vs {team2} → Winner: {winner}")
        winners.append(winner)
    return winners

def build_next_round(teams):
    return [(teams[i], teams[i+1]) for i in range(0, len(teams), 2)]

def simulate_tournament(initial_round, verbose=False):
    round_number = 1
    matches = initial_round
    all_rounds = [matches]
    winners_by_round = []

    while len(matches) > 1:
        winners = simulate_round(matches)
        winners_by_round.append(winners)
        matches = build_next_round(winners)
        all_rounds.append(matches)
        round_number += 1

    final_winner = matches[0][0]
    if verbose:
        print(f"\n🏆 Champion: {final_winner}")
    return final_winner, winners_by_round

if __name__ == "__main__":
    demo_round = [
        ("Brazil", "Germany"),
        ("Argentina", "France"),
        ("England", "Netherlands"),
        ("Spain", "Croatia"),
        ("Portugal", "USA"),
        ("Italy", "Switzerland"),
        ("Japan", "South Korea"),
        ("Uruguay", "Mexico")
    ]
    simulate_tournament(demo_round, verbose=True)


