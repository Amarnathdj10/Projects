import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

# ---------------- STYLING ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Poppins:wght@300;400&display=swap');
/* Dark overlay */
.stApp:before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.65);
    z-index: -1;
}

/* Fonts */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: white;
}

/* Title */
.title {
    font-family: 'Playfair Display', serif;
    text-align: center;
    font-size: 50px;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
    color: #ddd;
}

/* Card */
.card {
    background: rgba(255,255,255,0.1);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Button */
.stButton>button {
    background-color: #8B0000;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    background-color: #a30000;
}

/* Inputs */
label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🍷 Wine Quality Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered wine analysis system</div>', unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

with st.form('wine_form'):
    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
        volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.5)
        citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
        residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 2.0)
        chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.05)
        free_sulfur = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 30.0)

    with col2:
        total_sulfur = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 100.0)
        density = st.number_input("Density", 0.9900, 1.0100, 0.9950)
        pH = st.number_input("pH", 2.5, 4.5, 3.2)
        sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.5)
        alcohol = st.number_input("Alcohol", 5.0, 20.0, 10.0)
        wine_type = st.selectbox("Wine Type", ["Red", "White"])

    predict_btn = st.form_submit_button("Predict Quality 🍷")

st.markdown("</div>", unsafe_allow_html=True)

wine_type = 0 if wine_type == "Red" else 1

# ---------------- PREDICTION ----------------
if predict_btn:

    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur,
                            total_sulfur, density, pH, sulphates,
                            alcohol, wine_type]])

    st.write("### Model input (raw)", input_data.tolist())
    input_scaled = scaler.transform(input_data)
    st.write("### Model input (scaled)", input_scaled.tolist())

    prediction = model.predict(input_scaled)[0]

    labels = {0: "Bad ❌", 1: "Average ⚖️", 2: "Good ✅"}
    result = labels[prediction]

    # ---------- RESULT CARD ----------
    st.markdown('<div class="card" style="text-align:center; margin-top:20px;">', unsafe_allow_html=True)

    st.subheader("Prediction Result")
    st.markdown(f"<h2>{result}</h2>", unsafe_allow_html=True)

    # ---------- PROBABILITY ----------
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)[0]

        st.write("### Confidence Breakdown")

        st.progress(float(probs[0]))
        st.write(f"Bad ❌: {probs[0]*100:.2f}%")

        st.progress(float(probs[1]))
        st.write(f"Average ⚖️: {probs[1]*100:.2f}%")

        st.progress(float(probs[2]))
        st.write(f"Good ✅: {probs[2]*100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- EXPLANATION ----------
    st.markdown('<div class="card" style="margin-top:20px;">', unsafe_allow_html=True)

    st.subheader("Why this prediction?")

    reasons = []

    if alcohol > 11:
        reasons.append("✅ Higher alcohol improves quality")
    else:
        reasons.append("⚠️ Low alcohol reduces quality")

    if volatile_acidity < 0.4:
        reasons.append("✅ Low volatile acidity is good")
    else:
        reasons.append("⚠️ High volatile acidity harms quality")

    if sulphates > 0.6:
        reasons.append("✅ Good sulphate levels help stability")
    else:
        reasons.append("⚠️ Low sulphates may reduce quality")

    if density < 0.995:
        reasons.append("✅ Lower density indicates better balance")
    else:
        reasons.append("⚠️ Higher density may reduce quality")

    for r in reasons:
        st.write(r)

    st.markdown("</div>", unsafe_allow_html=True)