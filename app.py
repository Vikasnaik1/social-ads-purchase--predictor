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
    margin-bottom: 2rem;
}
.card {
    background-color: #0d1017;
    border: 1px solid #1c2332;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4f66;
    font-weight: 600;
    margin-bottom: 1rem;
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
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d4f66;
    margin-bottom: 0.35rem;
}
.stat-item-value {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e2e8f0;
}
.result-buy {
    background-color: #071a10;
    border: 1px solid #1a6b3a;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-nobuy {
    background-color: #1a0707;
    border: 1px solid #6b1a1a;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-desc {
    font-size: 0.83rem;
    color: #4a5568;
}
.badge {
    display: inline-block;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    margin-bottom: 1rem;
}
.badge-buy   { background-color:#0d3320; color:#34d174; border:1px solid #1a6b3a; }
.badge-nobuy { background-color:#2d0d0d; color:#f87171; border:1px solid #6b1a1a; }
.bar-wrap {
    height: 6px;
    background-color: #1c2332;
    border-radius: 99px;
    margin: 1.2rem 0 0.4rem;
    overflow: hidden;
}
.bar-fill-buy   { height:100%; border-radius:99px; background: linear-gradient(90deg,#1a6b3a,#34d174); }
.bar-fill-nobuy { height:100%; border-radius:99px; background: linear-gradient(90deg,#6b1a1a,#f87171); }
.bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #3d4f66;
}
.stButton > button {
    background-color: #1a6b3a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    border: 1px solid #1a6b3a !important;
    border-radius: 8px !important;
    height: 3em !important;
    width: 100% !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background-color: #22874a !important;
    border-color: #22874a !important;
}
.info-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.info-table td {
    padding: 0.5rem 0.4rem;
    border-bottom: 1px solid #1c2332;
}
.info-table td:first-child {
    color: #3d4f66;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    width: 45%;
}
.info-table td:last-child {
    color: #8899aa;
    font-weight: 500;
}
.warning-box {
    background-color: #1a1400;
    border: 1px solid #5a4200;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.8rem;
    color: #7a6030;
    line-height: 1.7;
    margin-bottom: 1rem;
}
.warning-box strong { color: #c8a040; }
.placeholder-box {
    border: 1px dashed #1c2332;
    border-radius: 10px;
    padding: 3rem 2rem;
    text-align: center;
}
.ph-title { font-size:0.9rem; font-weight:600; color:#283040; margin-bottom:0.3rem; }
.ph-sub   { font-size:0.78rem; color:#1c2332; }
hr { border-color: #1c2332 !important; margin: 1.5rem 0 !important; }
[data-testid="stSlider"] > div > div > div > div {
    background-color: #1a6b3a !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open("model9.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

SALARY_THRESHOLD = 103000


with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.8rem;'>
        <div style='font-size:1rem;font-weight:700;color:#e2e8f0;'>Purchase Predictor</div>
        <div style='font-size:0.65rem;color:#3d4f66;letter-spacing:0.12em;text-transform:uppercase;margin-top:3px;'>Social Ads Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div style='font-size:0.65rem;letter-spacing:0.1em;color:#3d4f66;margin-bottom:1rem;text-transform:uppercase;font-weight:600;'>Customer Profile</div>", unsafe_allow_html=True)

    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    age    = st.slider("Age", min_value=18, max_value=75, value=45, step=1)
    salary = st.number_input("Estimated Salary ($)", min_value=10000, max_value=500000, value=110000, step=1000)

    st.markdown("---")

    st.markdown("""
    <table class='info-table'>
        <tr><td>Model</td><td>SVC</td></tr>
        <tr><td>Kernel</td><td>RBF</td></tr>
        <tr><td>Features</td><td>Gender, Age, Salary</td></tr>
        <tr><td>Buy threshold</td><td>Salary &gt; $103,000</td></tr>
        <tr><td>Version</td><td>1.0.0</td></tr>
    </table>
    """, unsafe_allow_html=True)


st.markdown("<h1>Customer Purchase Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Predict whether a customer will purchase a product based on their profile.</div>", unsafe_allow_html=True)

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

    st.markdown(f"""
    <div class='warning-box'>
        <strong>Important — How This Model Predicts</strong><br>
        This SVC model was trained on unscaled data with a very small gamma value,
        which means <strong>Estimated Salary is the dominant feature</strong>.
        Age and Gender have minimal effect on the prediction.<br><br>
        Customers with salary <strong>above $103,000</strong> are predicted to buy.<br>
        Customers with salary <strong>below $103,000</strong> are predicted not to buy.<br><br>
        Current salary <strong>${salary:,}</strong> is
        <strong>{'above' if salary >= SALARY_THRESHOLD else 'below'}</strong>
        the threshold — result will be
        <strong>{'Will Buy' if salary >= SALARY_THRESHOLD else 'Will Not Buy'}</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='card-title'>What the result means</div>
        <div style='font-size:0.82rem;color:#4a5568;line-height:1.8;'>
            <strong style='color:#34d174;'>Will Buy</strong> — The customer profile matches
            historical buyers from the Social Network Ads dataset.<br>
            <strong style='color:#f87171;'>Will Not Buy</strong> — The customer profile does
            not match typical buyer patterns. Consider targeting with a different offer.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    run = st.button("RUN PREDICTION")

    if run:
        gender_val = 1 if gender == "Male" else 0
        df_input   = pd.DataFrame(
            [[gender_val, age, salary]],
            columns=["Gender", "Age", "EstimatedSalary"]
        )

        prediction     = model.predict(df_input)[0]
        decision_score = model.decision_function(df_input)[0]

        confidence_pct = min(int((abs(decision_score) / 3.0) * 100), 99)
        confidence_pct = max(confidence_pct, 5)

        if prediction == 1:
            st.markdown(f"""
            <div class='result-buy'>
                <div class='badge badge-buy'>Prediction Result</div>
                <div class='result-title' style='color:#34d174;'>Will Buy</div>
                <div class='result-desc'>This customer is likely to purchase the product.</div>
                <div class='bar-wrap'>
                    <div class='bar-fill-buy' style='width:{confidence_pct}%;'></div>
                </div>
                <div class='bar-labels'>
                    <span>Model Confidence</span>
                    <span>{confidence_pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class='result-nobuy'>
                <div class='badge badge-nobuy'>Prediction Result</div>
                <div class='result-title' style='color:#f87171;'>Will Not Buy</div>
                <div class='result-desc'>This customer is unlikely to purchase right now.</div>
                <div class='bar-wrap'>
                    <div class='bar-fill-nobuy' style='width:{confidence_pct}%;'></div>
                </div>
                <div class='bar-labels'>
                    <span>Model Confidence</span>
                    <span>{confidence_pct}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>Prediction Details</div>
            <table class='info-table'>
                <tr><td>Gender</td><td>{gender}</td></tr>
                <tr><td>Age</td><td>{age}</td></tr>
                <tr><td>Salary</td><td>${salary:,}</td></tr>
                <tr><td>Decision Score</td><td>{decision_score:.4f}</td></tr>
                <tr><td>Raw Output Class</td><td>{int(prediction)}</td></tr>
                <tr><td>Salary vs Threshold</td><td>${salary:,} vs $103,000</td></tr>
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
<div style='display:flex;justify-content:space-between;font-size:0.65rem;color:#1c2332;letter-spacing:0.05em;'>
    <span>Purchase Predictor v1.0.0</span>
    <span>SVC · RBF Kernel · Social Network Ads</span>
    <span>Deployed on Streamlit</span>
</div>
""", unsafe_allow_html=True)
