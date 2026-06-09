import pathlib

import pandas as pd
import streamlit as st
from predictor import load_saved_model, make_prediction

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "worldcup_predictor.joblib"


def _score_from_prediction(pred: str, probs: dict) -> str:
    """Choose a simple scoreline heuristic based on predicted class and its probability.

    This is a lightweight heuristic (no expected-goals model): stronger probabilities
    get larger margins.
    """
    p = float(probs.get(pred, 0))
    if pred == "Draw":
        if p > 0.75:
            return "1-1"
        if p > 0.5:
            return "0-0"
        return "1-1"
    if pred == "Home Win":
        if p > 0.75:
            return "3-0"
        if p > 0.55:
            return "2-0"
        if p > 0.45:
            return "2-1"
        return "1-0"
    # Away Win
    if p > 0.75:
        return "0-3"
    if p > 0.55:
        return "0-2"
    if p > 0.45:
        return "1-2"
    return "0-1"


def main() -> None:
    st.set_page_config(page_title="World Cup Outcome Predictor", layout="centered")
    st.title("World Cup Match Outcome Predictor")
    st.write(
        "Predict whether the home team will win, the away team will win, or the match will draw. "
        "Train the model first with `python train_model.py`, then run this app."
    )

    if not MODEL_PATH.exists():
        st.warning("No trained model found. Run `python train_model.py` first.")
        return

    model_data = load_saved_model(MODEL_PATH)
    home_teams = sorted(model_data["le_home"].classes_)
    away_teams = sorted(model_data["le_away"].classes_)

    with st.form(key="prediction_form"):
        home_team = st.selectbox("Home team", home_teams)
        away_team = st.selectbox("Away team", away_teams, index=1)
        match_date = st.date_input("Match date", value=pd.Timestamp.today().date())
        neutral = st.checkbox("Neutral venue", value=False)
        submitted = st.form_submit_button("Predict outcome")

    if submitted:
        if home_team == away_team:
            st.error("Home team and away team must be different.")
            return

        prediction, probabilities = make_prediction(
            model_data,
            home_team,
            away_team,
            neutral,
            pd.Timestamp(match_date),
        )

        st.markdown(f"### Predicted outcome: **{prediction}**")
        st.write(
            "Probability breakdown (higher means more likely):"
        )
        st.write(
            {
                label: f"{prob*100:.1f}%"
                for label, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            }
        )

        st.caption("The model is trained on historic FIFA World Cup results from the Kaggle dataset.")

    # Group predictions section
    with st.expander("Generate group-stage forecasts"):
        st.write("Click the button to generate fixture predictions and final group tables using the trained model.")
        if st.button("Generate group forecasts"):
            model_data = load_saved_model(MODEL_PATH)
            # default groups (use dataset labels mapping)
            raw_groups = {
                "Group A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
                "Group B": ["Canada", "Bosnia & Herzegovina", "Qatar", "Switzerland"],
                "Group C": ["Brazil", "Morocco", "Haiti", "Scotland"],
                "Group D": ["USA", "Paraguay", "Australia", "Türkiye"],
                "Group E": ["Germany", "Curaçao", "Côte d’Ivoire", "Ecuador"],
                "Group F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
                "Group G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
                "Group H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
                "Group I": ["France", "Senegal", "Iraq", "Norway"],
                "Group J": ["Argentina", "Algeria", "Austria", "Jordan"],
                "Group K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
                "Group L": ["England", "Croatia", "Ghana", "Panama"],
            }
            synonyms = {
                "Korea Republic": "South Korea",
                "USA": "United States",
                "Türkiye": "Turkey",
                "Côte d’Ivoire": "Ivory Coast",
                "Cabo Verde": "Cape Verde",
                "IR Iran": "Iran",
                "Czechia": "Czech Republic",
                "Bosnia & Herzegovina": "Bosnia and Herzegovina",
                "Congo DR": "DR Congo",
            }

            def normalize(team):
                return synonyms.get(team, team)

            from itertools import combinations

            for group_name, teams in raw_groups.items():
                norm = [normalize(t) for t in teams]
                st.subheader(group_name)
                # build fixtures
                fixtures = list(combinations(norm, 2))
                # table accumulator
                table = {t: {"P": 0, "W": 0, "D": 0, "L": 0, "Pts": 0} for t in norm}
                for home, away in fixtures:
                    try:
                        pred, probs = make_prediction(model_data, home, away, neutral=True, match_date=pd.Timestamp.today())
                    except Exception as e:
                        st.error(f"Prediction error for {home} vs {away}: {e}")
                        pred = "Draw"
                        probs = {"Draw": 1.0}

                    # display predicted score alongside the outcome and probabilities
                    score = _score_from_prediction(pred, probs)
                    probs_text = ", ".join(f"{k}: {v*100 if isinstance(v, float) else v}" for k, v in probs.items())
                    st.write(f"{home} {score} {away} — {pred} — {probs_text}")

                    if pred == "Home Win":
                        table[home]["W"] += 1
                        table[away]["L"] += 1
                        table[home]["Pts"] += 3
                    elif pred == "Away Win":
                        table[away]["W"] += 1
                        table[home]["L"] += 1
                        table[away]["Pts"] += 3
                    else:
                        table[home]["D"] += 1
                        table[away]["D"] += 1
                        table[home]["Pts"] += 1
                        table[away]["Pts"] += 1
                    table[home]["P"] += 1
                    table[away]["P"] += 1

                # show final table
                df_table = pd.DataFrame.from_dict(table, orient="index")
                df_table = df_table.sort_values(["Pts", "W", "D"], ascending=[False, False, False])
                st.table(df_table)


if __name__ == "__main__":
    main()
