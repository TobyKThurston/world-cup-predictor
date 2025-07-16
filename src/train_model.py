import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Load cleaned data
data = pd.read_csv("data/processed/cleaned_matches_extended.csv")


# Encode team names
le_team = LabelEncoder()

# Combine both columns to get all unique team names
all_teams = pd.concat([data['Home Team Name'], data['Away Team Name']]).unique()

le_team.fit(all_teams)

data['Home Team Encoded'] = le_team.transform(data['Home Team Name'])
data['Away Team Encoded'] = le_team.transform(data['Away Team Name'])


# Feature engineering: goal difference so far (very simple baseline feature)
data['Goal Diff'] = data['Home Team Goals'] - data['Away Team Goals']

# Features and target
X = data[['Home Team Encoded', 'Away Team Encoded']]
y = data['Result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and encoder
joblib.dump(model, "src/model.pkl")
joblib.dump(le_team, "src/label_encoder.pkl")


