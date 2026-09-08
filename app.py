import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model


# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------
# Custom CSS — colorful, friendly dashboard theme
# ----------------------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #FFF8F0 0%, #FDF2F8 50%, #F0F9FF 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6C5CE7 0%, #A29BFE 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stNumberInput input {
        color: #2D3436 !important;
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] button {
        color: #2D3436 !important;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(90deg, #FF9A8B 0%, #FF6A88 55%, #FF99AC 100%);
        padding: 28px 32px;
        border-radius: 20px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(255, 106, 136, 0.25);
    }
    .hero-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.1rem;
    }
    .hero-banner p {
        color: #FFF5F5;
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }

    /* Metric-style cards */
    .info-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border-left: 6px solid #6C5CE7;
        height: 100%;
    }
    .info-card h4 {
        margin: 0 0 4px 0;
        color: #636E72;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .info-card p {
        margin: 0;
        color: #2D3436;
        font-size: 1.4rem;
        font-weight: 700;
    }

    /* Result banners */
    .result-positive {
        background: linear-gradient(90deg, #FF7675 0%, #FD79A8 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 16px;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(253, 121, 168, 0.35);
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(90deg, #55EFC4 0%, #00B894 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 16px;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 6px 18px rgba(0, 184, 148, 0.35);
        text-align: center;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(90deg, #6C5CE7 0%, #A29BFE 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.35);
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #5A4BD1 0%, #8C84F5 100%);
        color: white;
    }

    /* Section headers */
    .section-title {
        color: #2D3436;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Load model and scaler
# ----------------------------
@st.cache_resource
def load_files():
    model = load_model("diabetes_mlp.keras")
    scaler = joblib.load("diabetes_scaler.pkl")
    return model, scaler


model, scaler = load_files()


# ----------------------------
# Hero banner
# ----------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🩺 Diabetes Prediction Dashboard</h1>
        <p>Enter patient details in the sidebar to estimate diabetes likelihood.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Sidebar — patient inputs
# ----------------------------
st.sidebar.markdown("## 🧾 Patient Details")
st.sidebar.markdown("Fill in the values below, then hit **Predict**.")
st.sidebar.markdown("---")

gender = st.sidebar.radio("⚧ Gender", options=["Female", "Male"], horizontal=True)
pregnancies = st.sidebar.number_input(
    "🤰 Pregnancies",
    min_value=0,
    value=1,
    disabled=(gender == "Male"),
    help="Not applicable for male patients.",
)
glucose = st.sidebar.number_input("🍬 Glucose", min_value=0.0, value=120.0)
blood_pressure = st.sidebar.number_input("💓 Blood Pressure", min_value=0.0, value=70.0)
skin_thickness = st.sidebar.number_input("📏 Skin Thickness", min_value=0.0, value=20.0)
insulin = st.sidebar.number_input("💉 Insulin", min_value=0.0, value=80.0)
bmi = st.sidebar.number_input("⚖️ BMI", min_value=0.0, value=25.0)
age = st.sidebar.number_input("🎂 Age", min_value=1, value=30)

# The trained model expects a Diabetes Pedigree Function value as one of its
# 8 input features, but this field is no longer collected from the user.
# We pass the dataset's typical average so the model still receives a valid,
# reasonable input in that slot.
diabetes_pedigree = 0.4720

if gender == "Male":
    pregnancies = 0

st.sidebar.caption(
    "ℹ️ Gender is shown for patient context only — the trained model does "
    "not currently use it as a prediction feature."
)

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("🔮 Predict Diabetes", use_container_width=True)


# ----------------------------
# Main area — quick overview cards
# ----------------------------
st.markdown('<p class="section-title">📋 Entered Values</p>', unsafe_allow_html=True)

card_cols = st.columns(5)
card_data = [
    ("Gender", gender),
    ("Glucose", f"{glucose:.0f}"),
    ("Blood Pressure", f"{blood_pressure:.0f}"),
    ("BMI", f"{bmi:.1f}"),
    ("Age", f"{age}"),
]
for col, (label, value) in zip(card_cols, card_data):
    with col:
        st.markdown(
            f"""
            <div class="info-card">
                <h4>{label}</h4>
                <p>{value}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# ----------------------------
# Prediction
# ----------------------------
if predict_clicked:

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    ]])

    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled, verbose=0)[0][0]
    prob_pct = probability * 100

    st.markdown('<p class="section-title">🎯 Prediction Result</p>', unsafe_allow_html=True)

    result_col, gauge_col = st.columns([1, 1.2])

    with result_col:
        if probability >= 0.5:
            st.markdown(
                f'<div class="result-positive">⚠️ Prediction: Diabetic<br>'
                f'Probability: {prob_pct:.2f}%</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-negative">✅ Prediction: Non-Diabetic<br>'
                f'Probability: {prob_pct:.2f}%</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.caption(
            "This estimate is based on the patient details provided and the "
            "trained model's output probability."
        )

    with gauge_col:
        gauge_color = "#FF6A88" if probability >= 0.5 else "#00B894"
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#636E72"},
                    "bar": {"color": gauge_color},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "#DFF9F0"},
                        {"range": [50, 100], "color": "#FFE3E8"},
                    ],
                    "threshold": {
                        "line": {"color": "#2D3436", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            )
        )
        fig.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#2D3436"},
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Fill in the patient details in the sidebar and click **Predict Diabetes** to see results here.")


# ----------------------------
# Disclaimer
# ----------------------------
st.divider()
st.caption(
    "For academic and educational purposes only. "
    "This application is not a medical diagnostic tool."
)