from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------
# LOAD MODEL + DATA
# ---------------------------

model = joblib.load("model.pickle")

df = pd.read_csv("ratings.csv")
df.columns = df.columns.str.strip()

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["OpponentRank"] = pd.to_numeric(df["OpponentRank"], errors="coerce")

df = df.sort_values(by=["Player", "Date"])
df = df.dropna(subset=["Rating"])

max_rank = df["OpponentRank"].max(skipna=True)
df["OpponentRank"] = df["OpponentRank"].fillna(max_rank + 10)

max_rank = df["OpponentRank"].max()
min_rank = df["OpponentRank"].min()

# ---------------------------
# FIXTURES
# ---------------------------

fixtures = {
    "Neymar": {
        "opponent": "Reims",
        "opponent_rank": None,
        "home": 1,
        "date": "2019-09-26"
    },
    "N'golo Kante": {
        "opponent": "Tottenham",
        "opponent_rank": 17,
        "home": 0,
        "date": "2018-11-24"
    },
    "Sergio Ramos": {
        "opponent": "Rayo Vallecano",
        "opponent_rank": None,
        "home": 1,
        "date": "2018-12-15"
    }
}

# ---------------------------
# ROUTES
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/fixture", methods=["POST"])
def get_fixture():
    data = request.get_json()
    player = data.get("player")

    if player not in fixtures:
        return jsonify({"error": "Player not found"}), 400

    return jsonify(fixtures[player])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        player = data.get("player")

        if player not in fixtures:
            return jsonify({"error": "Player not found"}), 400

        player_df = df[df["Player"] == player].sort_values("Date")

        if len(player_df) < 5:
            return jsonify({"error": "Not enough data"}), 400

        # ---- Feature Engineering ----

        form = player_df["Rating"].tail(5).mean()
        baseline = player_df["Rating"].mean()
        minutes = player_df["MinutesPlayed"].tail(5).mean()

        last_match_date = player_df["Date"].iloc[-1]
        fixture_date = pd.to_datetime(fixtures[player]["date"])
        rest_days = (fixture_date - last_match_date).days

        position = player_df["Position"].iloc[-1]
        position_forward = 1 if position == "Forward" else 0
        position_midfielder = 1 if position == "Midfielder" else 0

        opponent_rank = fixtures[player]["opponent_rank"]
        if opponent_rank is None:
            opponent_rank = max_rank + 10

        opponent_strength = 1 - (
            (opponent_rank - min_rank) /
            (max_rank - min_rank)
        )

        X_input = np.array([[ 
            fixtures[player]["home"],
            minutes,
            form,
            rest_days,
            opponent_strength,
            baseline,
            position_forward,
            position_midfielder
        ]])

        prediction = model.predict(X_input)[0]

        return jsonify({
            "predicted_rating": round(float(prediction), 2)
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()