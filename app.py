import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Purchase Predictor",
    page_icon="https://img.icons8.com/fluency/96/artificial-intelligence.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0c14;
    color: #c9d1e0;
}

[data-testid="stSidebar"] {
    background-color: #0d1017;
    border-right: 1px solid #1c2332;
}

[data-testid="stSidebar"] label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a5568;
    font-weight: 500;
}

.main .block-container {
    background-color: #0a0c14;
    padding: 2.5rem 3rem;
    max-width: 1200px;
}

h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.page-sub {
    font-size: 0.88rem;
    color: #3d4f66;
    margin-bottom: 2.5rem;
    font-weight: 400;
}

.card {
    background-color: #0d1017;
    border: 1px solid #1c2332;
    border-radius: 10px;
    padding: 1.5rem;
}

.card-title {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4f66;
    font-weight: 600;
    margin-bottom: 1.2rem;
}

.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}

.stat-item {
    flex: 1;
    background-color: #111520;
    border: 1px solid #1c2332;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}

.stat-item-label {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4f66;
    margin-bottom: 0.35rem;
}

.stat-item-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
}

.result-buy {
    background-color: #071a10;
    border: 1px solid #1a6b3a;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
}

.result-nobuy {
    background-color: #1a0707;
    border: 1px solid #6b1a1a;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
}

.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    letter-spacing: -0.01em;
}

.result-desc {
    font-size: 0.83rem;
    color: #4a5568;
}

.badge-buy {
    display: inline-block;
    background-color: #0d3320;
    color: #34d174;
    border: 1px solid #1a6b3a;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    margin-bottom: 1rem;
}

.badge-nobuy {
    display: inline-block;
    background-color: #2d0d0d;
    color: #f87171;
    border: 1px solid #6b1a1a;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    margin-bottom: 1rem;
}

.decision-bar-wrap {
    height: 6px;
    background-color: #1c2332;
    border-radius: 99px;
    margin: 1rem 0 0.4rem;
    overflow: hidden;
}

.decision-bar-fill-buy {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #1a6b3a, #34d174);
}

.decision-bar-fill-nobuy {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #6b1a1a, #f87171);
}

.decision-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #3d4f66;
}

.stButton > button {
    background-color: #1a6b3a;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    border: 1px solid #1a6b3a;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    transition: background-color 0.2s;
    text-transform: uppercase;
}

.stButton > button:hover {
    background-color: #22874a;
    border-color: #22874a;
}

.info-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 0.5rem;
}

.info-table td {
    padding: 0.55rem 0.5rem;
    border-bottom: 1px solid #1c2332;
    color: #4a5568;
}

.info-table td:first-child {
    color: #3d4f66;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    width: 40%;
}

.info-table td:last-child {
    color: #8899aa;
    font-weight: 500;
}

.placeholder-box {
    border: 1px dashed #1c2332;
    border-radius: 10px;
    padding: 3rem 2rem;
    text-align: center;
    color: #232d3d;
}

.placeholder-box .ph-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #283040;
    margin-bottom: 0.3rem;
}

.placeholder-box .ph-sub {
    font-size: 0.78rem;
    color: #1c2332;
}

hr { border-color: #1c2332 !important; margin: 1.5rem 0 !important; }

[data-testid="stSlider"] > div > div > div > div {
    background-color: #1a6b3a !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open("model8.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()


with st.sidebar:
    st.markdown("""
    <div style='padding: 1.2rem 0 0.8rem;'>
        <div style='font-size:1rem; font-weight:700; color:#e2e8f0; letter-spacing:-0.01em;'>Purchase Predictor</div>
        <div style='font-size:0.68rem; color:#3d4f66; letter-spacing:0.12em; text-transform:uppercase; margin-top:3px;'>Social Ads Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div style='font-size:0.68rem;letter-spacing:0.1em;color:#3d4f66;margin-bottom:1rem;text-transform:uppercase;font-weight:600;'>Customer Profile</div>", unsafe_allow_html=True)

    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    age = st.slider("Age", min_value=18, max_value=75, value=35, step=1)
    salary = st.number_input("Estimated Salary ($)", min_value=10000, max_value=500000, value=60000, step=1000)

    st.markdown("---")

    st.markdown("""
    <table class='info-table'>
        <tr><td>Model</td><td>SVC</td></tr>
        <tr><td>Kernel</td><td>RBF</td></tr>
        <tr><td>Features</td><td>Gender, Age, Salary</td></tr>
        <tr><td>Output</td><td>Buy / Not Buy</td></tr>
        <tr><td>Version</td><td>1.0.0</td></tr>
    </table>
    """, unsafe_allow_html=True)


st.markdown("<h1>Customer Purchase Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Determine whether a customer will purchase based on demographics and estimated salary.</div>", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown(f"""
    <div class='stat-row'>
        <div class='stat-item'>
            <div class='stat-item-label'>Gender</div>
            <div class='stat-item-value'>{gender}</div>
        </div>
        <div class='stat-item'>
            <div class='stat-item-label'>Age</div>
            <div class='stat-item-value'>{age}</div>
        </div>
        <div class='stat-item'>
            <div class='stat-item-label'>Est. Salary</div>
            <div class='stat-item-value'>${salary:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card' style='margin-top:0.5rem;'>
        <div class='card-title'>How Predictions Work</div>
        <div style='font-size:0.82rem; color:#4a5568; line-height:1.8;'>
            The model analyzes <strong style='color:#5a6f85;'>Gender</strong>,
            <strong style='color:#5a6f85;'>Age</strong>, and
            <strong style='color:#5a6f85;'>Estimated Salary</strong> to determine
            purchase likelihood based on patterns learned from historical ad campaign data.
            <br><br>
            <span style='color:#3d4f66;'><strong style='color:#5a6f85;'>Will Buy</strong> — customer is likely to convert.</span><br>
            <span style='color:#3d4f66;'><strong style='color:#5a6f85;'>Will Not Buy</strong> — low purchase intent detected.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:

    run = st.button("RUN PREDICTION")

    if run:
        gender_val = 1 if gender == "Male" else 0
        df_input = pd.DataFrame(
            [[gender_val, age, salary]],
            columns=["Gender", "Age", "EstimatedSalary"]
        )

        prediction = model.predict(df_input)[0]
        decision_score = model.decision_function(df_input)[0]

        score_magnitude = abs(decision_score)
        confidence_pct = min(int((score_magnitude / 3.0) * 100), 99)
        confidence_pct = max(confidence_pct, 5)

        if prediction == 1:
            st.markdown(f"""
            <div class='result-buy'>
                <div class='badge-buy'>Prediction Result</div>
                <div class='result-title' style='color:#34d174;'>Will Buy</div>
                <div class='result-desc'>This customer is likely to purchase the product.</div>
                <div class='decision-bar-wrap' style='margin-top:1.5rem;'>
                    <div class='decision-bar-fill-buy' style='width:{confidence_pct}%;'></div>
                </div>
                <div class='decision-label'>
                    <span>Model Confidence</span>
                    <span>{confidence_pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class='result-nobuy'>
                <div class='badge-nobuy'>Prediction Result</div>
                <div class='result-title' style='color:#f87171;'>Will Not Buy</div>
                <div class='result-desc'>This customer is unlikely to purchase right now.</div>
                <div class='decision-bar-wrap' style='margin-top:1.5rem;'>
                    <div class='decision-bar-fill-nobuy' style='width:{confidence_pct}%;'></div>
                </div>
                <div class='decision-label'>
                    <span>Model Confidence</span>
                    <span>{confidence_pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card' style='margin-top:1rem;'>
            <div class='card-title'>Prediction Details</div>
            <table class='info-table'>
                <tr><td>Gender</td><td>{gender}</td></tr>
                <tr><td>Age</td><td>{age}</td></tr>
                <tr><td>Salary</td><td>${salary:,}</td></tr>
                <tr><td>Decision Score</td><td>{decision_score:.4f}</td></tr>
                <tr><td>Raw Output</td><td>{int(prediction)}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='placeholder-box'>
            <div class='ph-title'>No Prediction Yet</div>
            <div class='ph-sub'>Set the customer profile and click Run Prediction</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='display:flex; justify-content:space-between; font-size:0.68rem; color:#1c2332; letter-spacing:0.06em;'>
    <span>Purchase Predictor v1.0.0</span>
    <span>SVC · RBF Kernel · Social Network Ads</span>
    <span>Deployed on Streamlit</span>
</div>
""", unsafe_allow_html=True)
