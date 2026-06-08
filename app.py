import pathlib

import pandas as pd
import streamlit as st
from predictor import load_saved_model, make_prediction

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "worldcup_predictor.joblib"


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


if __name__ == "__main__":
    main()
