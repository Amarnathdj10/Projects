"""
Golazo! — app.py
Serves the FastAPI data endpoints AND hosts the static frontend.

Run with:  uvicorn app:app --reload
Then open: http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os

app = FastAPI(title="Golazo! International Goalscorers API")

# ── DATA ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "goals.csv"))
df = df.sort_values(["Nation", "Goals"], ascending=[True, False])


# ── API ROUTES (must be declared BEFORE static mount) ────────────────────────

@app.get("/nations")
def get_nations():
    """Return all nations present in goals.csv, sorted alphabetically."""
    return {"nations": sorted(df["Nation"].unique().tolist())}


@app.get("/players/{nation}")
def get_players(nation: str):
    """Return top scorers for the given nation, ordered by goals descending."""
    data = df[df["Nation"] == nation]
    players = data[["Player", "Goals"]].to_dict(orient="records")
    return {"nation": nation, "players": players}


# ── STATIC FRONTEND (catch-all — must be mounted LAST) ──────────────────────
# Serves index.html, style.css, script.js, images/, etc.
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")