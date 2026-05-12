import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Executive Predictor",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #ff2b2b; border: none; }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error: {e}")

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png")
    st.title("Control Panel")
    st.markdown("---")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    salary = st.number_input("Estimated Salary ($)", min_value=10000, max_value=500000, value=50000)

st.title("🎯 Target Audience Intelligence")
st.markdown("#### Precision Prediction Engine")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("Configuration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gender", gender)
    c2.metric("Age", age)
    c3.metric("Salary", f"${salary:,}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Action")
    gender_val = 1 if gender == "Male" else 0
    input_data = np.array([[gender_val, age, salary]])
    
    if st.button("RUN PREDICTION"):
        prediction = model.predict(input_data)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.success("### result: POSITIVE")
            st.balloons()
        else:
            st.warning("### result: NEGATIVE")

st.markdown("---")
st.caption("v1.0.0 | SVM RBF Kernel")
