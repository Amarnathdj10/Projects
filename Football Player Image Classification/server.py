import os
import pickle
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

class_names = ["Messi", "Ronaldo", "Neymar", "Mbappe", "Haaland"]

def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 100))
    img = img.flatten() / 255.0
    return img.reshape(1, -1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    processed = preprocess_image(file)
    prediction = model.predict(processed)[0]

    return jsonify({
        "prediction": class_names[prediction]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)