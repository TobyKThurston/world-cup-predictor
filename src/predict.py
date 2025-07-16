import joblib
import numpy as np

# Load model and encoder
model = joblib.load("src/model.pkl")
le_team = joblib.load("src/label_encoder.pkl")

def predict_match(home_team, away_team):
    # Encode team names
    try:
        home_encoded = le_team.transform([home_team])[0]
        away_encoded = le_team.transform([away_team])[0]
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create feature array
    features = np.array([[home_encoded, away_encoded]])

    # Predict
    result = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    print(f"Predicted Result: {result}")
    print(f"Probabilities: {model.classes_[0]}: {probs[0]:.2f}, {model.classes_[1]}: {probs[1]:.2f}, {model.classes_[2]}: {probs[2]:.2f}")

if __name__ == "__main__":
    # Example usage
    home = input("Enter home team: ")
    away = input("Enter away team: ")
    predict_match(home, away)

