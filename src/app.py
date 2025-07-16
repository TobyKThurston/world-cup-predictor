import streamlit as st
import joblib
import sys, os

# Local import fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from group_simulator import create_groups, simulate_all_groups
from build_knockout import generate_knockout_bracket
from simulator import simulate_tournament

st.set_page_config(page_title="World Cup 2026 Simulator", layout="wide")
st.title("🏆 World Cup 2026 ML Simulator")
st.markdown("Predict the full 2026 FIFA World Cup using a machine learning model trained on real match results.")

st.divider()
st.header("⚽️ Full Tournament Simulation")

# Preload model (forces Vercel to cache these later)
joblib.load("src/model.pkl")
joblib.load("src/label_encoder.pkl")

if st.button("🔁 Simulate World Cup"):
    # Group stage
    groups = create_groups()
    group_results, qualified_teams = simulate_all_groups(groups)

    # Knockout
    bracket = generate_knockout_bracket(qualified_teams)
    champion, rounds = simulate_tournament(bracket)

    # Champion display
    st.success(f"🏆 Predicted Champion: **{champion}**")

    # Display bracket
    round_names = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
    cols = st.columns(len(rounds))

    for i, round_data in enumerate(rounds):
        with cols[i]:
            st.markdown(f"#### {round_names[i]}")
            for match in round_data:
                matchup = match["match"]
                winner = match["winner"]
                prob = match["prob"]
                st.markdown(
                    f"🆚 {matchup}  \n→ 🏅 **{winner}** (prob: `{prob:.2f}`)", unsafe_allow_html=True
                )

