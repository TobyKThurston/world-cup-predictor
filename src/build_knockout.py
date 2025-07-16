import random

# Shuffle and pair up 32 teams
def generate_knockout_bracket(qualified_teams):
    random.shuffle(qualified_teams)
    return [(qualified_teams[i], qualified_teams[i+1]) for i in range(0, 32, 2)]

if __name__ == "__main__":
    teams = [f"Team{i}" for i in range(32)]
    bracket = generate_knockout_bracket(teams)
    for match in bracket:
        print(match)
