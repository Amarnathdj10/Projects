import os
import numpy as np
import cv2
import joblib
import json
import pywt
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model
model = joblib.load("model.pickle")

# Load class dictionary
with open("class_dictionary.json", "r") as f:
    class_dict = json.load(f)

class_dict_inv = {v: k for k, v in class_dict.items()}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H


def get_cropped_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        return roi_color
    return None


def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    cropped = get_cropped_image(img)
    if cropped is None:
        return None

    scaled_img = cv2.resize(cropped, (32, 32))
    im_har = w2d(cropped, 'db1', 2)
    scaled_img_har = cv2.resize(im_har, (32, 32))

    combined_img = np.vstack((
        scaled_img.reshape(32*32*3, 1),
        scaled_img_har.reshape(32*32, 1)
    ))

    return combined_img.reshape(1, -1).astype(float)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    processed = preprocess_image(file)

    if processed is None:
        return jsonify({"error": "No face detected"})

    prediction = model.predict(processed)[0]

    return jsonify({
        "prediction": class_dict_inv[prediction]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)