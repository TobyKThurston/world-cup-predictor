import random
import joblib
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.teams_2026 import teams_2026


model = joblib.load("src/model.pkl")
le_team = joblib.load("src/label_encoder.pkl")

# Randomly assign 48 teams to 12 groups of 4
def create_groups():
    teams = teams_2026.copy()
    random.shuffle(teams)
    groups = {}
    for i in range(12):
        group_name = f"Group {chr(65 + i)}"
        groups[group_name] = teams[i*4:(i+1)*4]
    return groups

# Simulate a single match using model probabilities
def simulate_match(team1, team2):
    try:
        t1 = le_team.transform([team1])[0]
        t2 = le_team.transform([team2])[0]
    except ValueError as e:
        raise ValueError(f"Unknown team in model: {e}")


    probs = model.predict_proba([[t1, t2]])[0]
    classes = model.classes_
    prob_dict = dict(zip(classes, probs))

    rand = random.random()
    if rand < prob_dict["HomeWin"]:
        return team1, 3, 0
    elif rand < prob_dict["HomeWin"] + prob_dict["AwayWin"]:
        return team2, 0, 3
    else:
        return None, 1, 1  # draw

# Simulate round-robin group play
def simulate_group(group_teams):
    scores = defaultdict(lambda: {"pts": 0, "gd": 0, "gf": 0})
    for i in range(4):
        for j in range(i+1, 4):
            team1, team2 = group_teams[i], group_teams[j]
            winner, pts1, pts2 = simulate_match(team1, team2)
            if winner == team1:
                goal_diff = random.randint(1, 3)
                scores[team1]["pts"] += pts1
                scores[team1]["gd"] += goal_diff
                scores[team1]["gf"] += goal_diff
                scores[team2]["gd"] -= goal_diff
            elif winner == team2:
                goal_diff = random.randint(1, 3)
                scores[team2]["pts"] += pts2
                scores[team2]["gd"] += goal_diff
                scores[team2]["gf"] += goal_diff
                scores[team1]["gd"] -= goal_diff
            else:  # draw
                goals = random.randint(0, 2)
                scores[team1]["pts"] += 1
                scores[team2]["pts"] += 1
                scores[team1]["gf"] += goals
                scores[team2]["gf"] += goals

    return sorted(group_teams, key=lambda t: (scores[t]["pts"], scores[t]["gd"], scores[t]["gf"]), reverse=True), scores

# Simulate all groups
def simulate_all_groups(groups):
    all_results = {}
    third_place_teams = []

    for name, teams in groups.items():
        if len(teams) != 4:
            print(f"❌ Error: {name} has {len(teams)} teams: {teams}")
        ranked, score_table = simulate_group(teams)
        all_results[name] = {
            "teams": ranked,
            "scores": score_table
        }
        # Save 3rd place team for later
        third = ranked[2]
        third_place_teams.append((third, score_table[third]))

    # Pick best 8 third-place teams
    third_place_teams.sort(key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]), reverse=True)
    best_third = [team for team, _ in third_place_teams[:8]]

    # Return top 2 from each group + best 8 third place
    qualified = []
    for group in all_results.values():
        qualified.extend(group["teams"][:2])  # top 2
    qualified.extend(best_third)

    return all_results, qualified  # 32 total

if __name__ == "__main__":
    groups = create_groups()
    results, knockout = simulate_all_groups(groups)
    print("Qualified for Knockout Stage:")
    for team in knockout:
        print(team)
