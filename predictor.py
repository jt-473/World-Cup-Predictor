import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "worldcup_predictor.joblib"

FEATURE_COLUMNS = [
    "home_team_id",
    "away_team_id",
    "neutral",
    "elo_diff",
    "home_form",
    "away_form",
]

TARGET_LABELS = ["Home Win", "Away Win", "Draw"]

# Only include these teams in training and the app
ALLOWED_TEAMS = {
    "Canada",
    "Mexico",
    "United States",
    "Australia",
    "Iran",
    "Iraq",
    "Japan",
    "Jordan",
    "South Korea",
    "Qatar",
    "Saudi Arabia",
    "Uzbekistan",
    "Algeria",
    "Cape Verde",
    "DR Congo",
    "Ivory Coast",
    "Egypt",
    "Ghana",
    "Morocco",
    "Senegal",
    "South Africa",
    "Tunisia",
    "Curaçao",
    "Haiti",
    "Panama",
    "Argentina",
    "Brazil",
    "Colombia",
    "Ecuador",
    "Paraguay",
    "Uruguay",
    "New Zealand",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Croatia",
    "Czech Republic",
    "England",
    "France",
    "Germany",
    "Netherlands",
    "Norway",
    "Portugal",
    "Scotland",
    "Spain",
    "Sweden",
    "Switzerland",
    "Turkey",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column name variants from different CSV sources.

    Ensures the dataframe contains `home_team_goal`, `away_team_goal`,
    `date`, and `neutral` columns used throughout the code.
    """
    df = df.copy()

    # Home goal candidates
    home_candidates = [
        "home_team_goal",
        "home_team_goals",
        "home_score",
        "home_goals",
        "home_team_score",
    ]
    for c in home_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "home_team_goal"})
            break

    # Away goal candidates
    away_candidates = [
        "away_team_goal",
        "away_team_goals",
        "away_score",
        "away_goals",
        "away_team_score",
    ]
    for c in away_candidates:
        if c in df.columns:
            df = df.rename(columns={c: "away_team_goal"})
            break

    # Normalize `date` column case-insensitively if needed
    if "date" not in df.columns:
        for c in df.columns:
            if c.lower() == "date":
                df = df.rename(columns={c: "date"})
                break

    # Ensure `neutral` exists
    if "neutral" not in df.columns:
        df["neutral"] = False

    return df


def load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = _normalize_columns(df)
    return df


def filter_world_cup(df: pd.DataFrame) -> pd.DataFrame:
    world_cup = df[df["tournament"].str.contains("World Cup", case=False, na=False)].copy()
    return world_cup if not world_cup.empty else df.copy()


def add_match_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["neutral"] = df["neutral"].astype(int)

    df["target"] = np.where(
        df["home_team_goal"] > df["away_team_goal"],
        "Home Win",
        np.where(
            df["home_team_goal"] < df["away_team_goal"],
            "Away Win",
            "Draw",
        ),
    )
    return df


def encode_teams(df: pd.DataFrame):
    # Fit a single encoder over all teams to ensure consistent ids
    teams = pd.Index(df["home_team"]).append(pd.Index(df["away_team"]))
    teams = teams.unique()
    le = LabelEncoder().fit(teams)
    df["home_team_id"] = le.transform(df["home_team"])
    df["away_team_id"] = le.transform(df["away_team"])
    return df, le, le


def _compute_elo_and_form(df: pd.DataFrame, k: int = 20, initial_elo: float = 1500.0, form_n: int = 5):
    """Compute rolling Elo and recent-form features for each match.

    Adds columns `home_elo`, `away_elo`, `elo_diff`, `home_form`, `away_form` to a copy of df.
    Returns (df_with_features, final_elos, mean_form_per_team)
    """
    df = df.sort_values("date").copy()
    from collections import defaultdict

    elos = defaultdict(lambda: initial_elo)
    recent = defaultdict(list)

    home_elos = []
    away_elos = []
    home_forms = []
    away_forms = []

    for _, row in df.iterrows():
        h = row["home_team"]
        a = row["away_team"]

        home_elos.append(elos[h])
        away_elos.append(elos[a])

        def form_score(team):
            lst = recent[team]
            return sum(lst[-form_n:]) if lst else 0

        home_forms.append(form_score(h))
        away_forms.append(form_score(a))

        # compute match result points
        if row.get("home_team_goal", 0) > row.get("away_team_goal", 0):
            ph, pa = 3, 0
            rh, ra = 1.0, 0.0
        elif row.get("home_team_goal", 0) < row.get("away_team_goal", 0):
            ph, pa = 0, 3
            rh, ra = 0.0, 1.0
        else:
            ph, pa = 1, 1
            rh, ra = 0.5, 0.5

        recent[h].append(ph)
        recent[a].append(pa)

        # Elo update
        expected_h = 1.0 / (1.0 + 10 ** ((elos[a] - elos[h]) / 400.0))
        expected_a = 1.0 - expected_h
        elos[h] = elos[h] + k * (rh - expected_h)
        elos[a] = elos[a] + k * (ra - expected_a)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["home_form"] = home_forms
    df["away_form"] = away_forms

    # final elos and mean form per team
    final_elos = {team: float(val) for team, val in elos.items()}
    mean_form = {team: float(sum(vals) / len(vals)) if vals else 0.0 for team, vals in recent.items()}

    return df, final_elos, mean_form


def build_training_data(df: pd.DataFrame):
    df = filter_world_cup(df)
    df = add_match_features(df)
    # compute Elo and recent-form features
    df, final_elos, mean_form = _compute_elo_and_form(df)
    # Keep only matches where both teams are in the allowed set
    if "home_team" in df.columns and "away_team" in df.columns:
        df = df[df["home_team"].isin(ALLOWED_TEAMS) & df["away_team"].isin(ALLOWED_TEAMS)].copy()

    if df.empty:
        raise ValueError(
            "No matches found for the configured ALLOWED_TEAMS.\n"
            "Make sure team names in the dataset exactly match the allowed team names."
        )

    df, le_home, le_away = encode_teams(df)
    X = df[FEATURE_COLUMNS]
    y = df["target"]
    return X, y, le_home, le_away, final_elos, mean_form


def train_and_save_model(csv_path: pathlib.Path, model_path: pathlib.Path = MODEL_PATH):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_data(csv_path)
    X, y, le_home, le_away, final_elos, mean_form = build_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nModel classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(
        {
            "model": model,
            "le_home": le_home,
            "le_away": le_away,
            "feature_columns": FEATURE_COLUMNS,
            "team_elos": final_elos,
            "team_mean_form": mean_form,
        },
        model_path,
    )
    print(f"Saved model to: {model_path}")
    return model_path


def load_saved_model(model_path: pathlib.Path = MODEL_PATH):
    return joblib.load(model_path)


def make_prediction(model_data, home_team: str, away_team: str, neutral: bool, match_date: pd.Timestamp):
    le_home = model_data["le_home"]
    le_away = model_data["le_away"]
    model = model_data["model"]

    if home_team not in le_home.classes_:
        raise ValueError(f"Home team not found in trained data: {home_team}")
    if away_team not in le_away.classes_:
        raise ValueError(f"Away team not found in trained data: {away_team}")

    # Build input row using the feature column names saved with the model so
    # predictions remain compatible if training features change.
    feature_cols = model_data.get("feature_columns", FEATURE_COLUMNS)
    values = {}
    for col in feature_cols:
        if col == "home_team_id":
            values[col] = int(le_home.transform([home_team])[0])
        elif col == "away_team_id":
            values[col] = int(le_away.transform([away_team])[0])
        elif col == "neutral":
            values[col] = int(neutral)
        elif col == "elo_diff":
            team_elos = model_data.get("team_elos", {})
            home_elo = team_elos.get(home_team, 1500.0)
            away_elo = team_elos.get(away_team, 1500.0)
            values[col] = float(home_elo - away_elo)
        elif col == "home_form":
            mean_form = model_data.get("team_mean_form", {})
            values[col] = float(mean_form.get(home_team, 0.0))
        elif col == "away_form":
            mean_form = model_data.get("team_mean_form", {})
            values[col] = float(mean_form.get(away_team, 0.0))
        else:
            raise ValueError(f"Unsupported feature column: {col}")

    row = pd.DataFrame([values])

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    proba_map = dict(zip(model.classes_, probabilities))
    return prediction, proba_map
