import streamlit as st
import pickle
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BuySignal AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #06080f;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1120 0%, #111827 100%);
    border-right: 1px solid #1e2a3a;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c8fa6;
}

/* ── Main background ── */
.main .block-container {
    background-color: #06080f;
    padding-top: 2rem;
}

/* ── Hero title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #4b5e72;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 2.5rem;
}

/* ── Stat cards ── */
.stat-grid {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    flex: 1;
    background: #0d1120;
    border: 1px solid #1a2336;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}
.stat-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4b5e72;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8eaf0;
}

/* ── Result cards ── */
.result-positive {
    background: linear-gradient(135deg, #052e1a 0%, #0a3d26 100%);
    border: 1px solid #16a34a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s ease-in-out infinite;
}
.result-negative {
    background: linear-gradient(135deg, #1a0505 0%, #2d0d0d 100%);
    border: 1px solid #dc2626;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(22,163,74,0.4); }
    50%       { box-shadow: 0 0 20px 6px rgba(22,163,74,0.15); }
}

.result-emoji  { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label  {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-desc   { font-size: 0.9rem; color: #94a3b8; }

/* ── Probability bar ── */
.prob-bar-wrap  { background: #1a2336; border-radius: 99px; height: 10px; margin: 0.8rem 0; overflow: hidden; }
.prob-bar-fill  { height: 100%; border-radius: 99px; transition: width 0.8s ease; }
.prob-label     { font-size: 0.78rem; color: #64748b; display: flex; justify-content: space-between; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    color: #06080f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 10px !important;
    height: 3.2em !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Info box ── */
.info-box {
    background: #0d1120;
    border: 1px solid #1a2336;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.5rem;
    font-size: 0.85rem;
    color: #64748b;
    line-height: 1.7;
}
.info-box strong { color: #94a3b8; }

/* ── Divider ── */
hr { border-color: #1a2336 !important; margin: 1.5rem 0 !important; }

/* ── Slider & input tweaks ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """
    Load model and (optionally) scaler.
    Supports two layouts:
      1. model.pkl  +  scaler.pkl   (recommended – best accuracy)
      2. model.pkl only             (scaling done internally if needed)
    """
    model, scaler = None, None
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️  `model.pkl` not found. Please upload it to the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️  Could not load model: {e}")
        st.stop()

    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        pass   # scaler is optional
    except Exception:
        pass

    return model, scaler

model, scaler = load_artifacts()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800;
                    background:linear-gradient(90deg,#38bdf8,#818cf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;'>
            BuySignal AI
        </div>
        <div style='font-size:0.72rem; color:#4b5e72; letter-spacing:0.15em; margin-top:2px;'>
            CUSTOMER INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;letter-spacing:0.15em;color:#4b5e72;margin-bottom:0.8rem;text-transform:uppercase;'>Customer Profile</div>", unsafe_allow_html=True)

    gender  = st.selectbox("Gender", ["Male", "Female"])
    age     = st.slider("Age", 18, 75, 35)
    salary  = st.number_input("Annual Salary (₹ / $)", min_value=10_000, max_value=500_000,
                               value=60_000, step=5_000)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#334155; line-height:1.6;'>
        Model: <span style='color:#4b5e72;'>SVM · RBF Kernel</span><br>
        Version: <span style='color:#4b5e72;'>1.0.0</span>
    </div>
    """, unsafe_allow_html=True)

# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>Customer Purchase Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Predict whether a customer will buy your product — powered by SVM</div>", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    # Stat cards
    gender_icon = "👨" if gender == "Male" else "👩"
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-label">Gender</div>
            <div class="stat-value">{gender_icon} {gender}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Age</div>
            <div class="stat-value">{age} yrs</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Salary</div>
            <div class="stat-value">₹{salary:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Insight box
    # Simple rule-of-thumb insight (not the model — just UX context)
    likely_buyer = (age >= 28) and (salary >= 60_000)
    insight_color = "#16a34a" if likely_buyer else "#dc2626"
    insight_icon  = "📈" if likely_buyer else "📉"
    insight_text  = "Profile matches typical buyer demographics." if likely_buyer \
                    else "Profile is outside typical high-conversion segments."

    st.markdown(f"""
    <div class='info-box'>
        {insight_icon} <strong>Quick Insight:</strong> {insight_text}<br>
        <br>
        <strong>How to read the result:</strong><br>
        🟢 <strong>Will Buy</strong> — High probability the customer purchases the product.<br>
        🔴 <strong>Won't Buy</strong> — Low purchase intent; consider a different offer or retargeting.
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    run = st.button("⚡  PREDICT NOW")

    if run:
        gender_val  = 1 if gender == "Male" else 0
        raw_input   = np.array([[gender_val, age, salary]], dtype=float)

        # ── Scaling ─────────────────────────────────────────────────────────
        # SVM-RBF MUST have scaled input; without a saved scaler we apply
        # a manual normalisation matching the typical Social-Network-Ads range.
        if scaler is not None:
            model_input = scaler.transform(raw_input)
        else:
            # Fallback: manual standardisation (approximate)
            # Gender: 0/1 → leave; Age: mean≈37, std≈10; Salary: mean≈70k, std≈34k
            age_scaled    = (age    - 37.0) / 10.0
            salary_scaled = (salary - 70000.0) / 34000.0
            model_input   = np.array([[gender_val, age_scaled, salary_scaled]])

        # ── Prediction ──────────────────────────────────────────────────────
        prediction = model.predict(model_input)[0]

        # Probability (SVM may or may not support predict_proba)
        try:
            proba     = model.predict_proba(model_input)[0]
            buy_prob  = proba[1]
            has_proba = True
        except AttributeError:
            has_proba = False
            buy_prob  = 1.0 if prediction == 1 else 0.0

        # ── Display result ───────────────────────────────────────────────────
        if prediction == 1:
            st.markdown(f"""
            <div class='result-positive'>
                <div class='result-emoji'>🎯</div>
                <div class='result-label' style='color:#4ade80;'>Will Buy!</div>
                <div class='result-desc'>This customer is likely to purchase your product.</div>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class='result-negative'>
                <div class='result-emoji'>🚫</div>
                <div class='result-label' style='color:#f87171;'>Won't Buy</div>
                <div class='result-desc'>This customer is unlikely to purchase right now.</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar
        if has_proba:
            pct      = int(buy_prob * 100)
            bar_col  = "#4ade80" if buy_prob >= 0.5 else "#f87171"
            st.markdown(f"""
            <div style='margin-top:1rem;'>
                <div class='prob-label'>
                    <span>Purchase Probability</span>
                    <span style='color:#e8eaf0; font-weight:600;'>{pct}%</span>
                </div>
                <div class='prob-bar-wrap'>
                    <div class='prob-bar-fill'
                         style='width:{pct}%; background:{bar_col};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Placeholder before prediction
        st.markdown("""
        <div style='border:1px dashed #1a2336; border-radius:14px; padding:2.5rem 1.5rem;
                    text-align:center; color:#2a3a52;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>🎯</div>
            <div style='font-family:Syne,sans-serif; font-size:1rem; font-weight:600;'>
                Ready to Predict
            </div>
            <div style='font-size:0.82rem; margin-top:0.3rem;'>
                Fill in the profile on the left and hit Predict Now
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='display:flex; justify-content:space-between; font-size:0.72rem; color:#2a3a52;'>
    <span>BuySignal AI · v1.0.0</span>
    <span>SVM · RBF Kernel · Social Network Ads Dataset</span>
    <span>Built with Streamlit</span>
</div>
""", unsafe_allow_html=True)
