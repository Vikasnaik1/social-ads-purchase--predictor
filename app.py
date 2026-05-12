import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Executive Predictor",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: #ffffff;
}

.block {
    background-color: #1e2130;
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid #ff4b4b;
}

.stButton>button {
    width: 100%;
    border-radius: 6px;
    height: 3em;
    background-color: #ff4b4b;
    color: white;
    border: none;
}

.stButton>button:hover {
    background-color: #ff2b2b;
}
</style>
""", unsafe_allow_html=True)

try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    st.error(str(e))
    st.stop()

st.title("Target Audience Intelligence")
st.markdown("Prediction system for customer purchase behavior")

with st.sidebar:
    st.title("Input Controls")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    salary = st.number_input("Salary", 10000, 500000, 50000)

gender_val = 1 if gender == "Male" else 0
input_data = np.array([[gender_val, age, salary]])

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("Input Summary")
    st.write("Gender:", gender)
    st.write("Age:", age)
    st.write("Salary:", salary)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Prediction")

    if st.button("Run Prediction"):
        try:
            prediction = model.predict(input_data)

            result = prediction[0]

            if result == 1:
                st.success("Positive")
            else:
                st.warning("Negative")

        except Exception as e:
            st.error(str(e))

st.markdown("---")
st.caption("SVM Model | Prediction System")
