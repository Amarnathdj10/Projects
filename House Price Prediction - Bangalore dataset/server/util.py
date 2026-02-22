import json
import pickle
from flask import app
import numpy as np
import os

__locations = None
__data_columns = None
__model = None

def get_location_names():
    return __locations

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0],2)

def load_saved_artifacts():
    print("Loading saved artifacts...")
    global __data_columns
    global __locations
    global __model

    # Get current directory of this file
    current_dir = os.path.dirname(__file__)

    # Build proper relative paths
    columns_path = os.path.join(current_dir, "artifacts", "columns.json")
    model_path = os.path.join(current_dir, "artifacts", "bangalore_home_prices_model.pickle")

    # Load columns
    with open(columns_path, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    # Load model
    with open(model_path, "rb") as f:
        __model = pickle.load(f)

    print("Loading artifacts done...")

# Load artifacts when module loads (for production)
load_saved_artifacts()

if __name__ == '__main__':
    app.run()