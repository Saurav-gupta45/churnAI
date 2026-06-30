import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ─── PAGE CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title="ChurnAI — Prediction Engine",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── BACKEND (unchanged) ─────────────────────────────
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ─── GLOBAL CSS ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0F172A !important;
    font-family: 'Fira Sans', sans-serif !important;
    color: #F8FAFC !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(245,158,11,0.12) 0%, transparent 60%),
                radial-gradient(ellipse 50% 50% at 85% 80%,  rgba(139,92,246,0.08) 0%, transparent 50%),
                #0F172A !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="block-container"] { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] { padding: 0 !important; }

/* ── Animated blobs ── */
.blob-container {
    position: fixed; inset: 0; z-index: 0;
    pointer-events: none; overflow: hidden;
}
.blob {
    position: absolute; border-radius: 50%;
    filter: blur(90px); opacity: 0.10;
    animation: blob-float linear infinite alternate;
}
.b1 { width: 600px; height: 600px; background: #F59E0B; top: -200px; left: -150px; animation-duration: 22s; }
.b2 { width: 400px; height: 400px; background: #8B5CF6; bottom: -100px; right: -100px; animation-duration: 17s; animation-delay: -8s; }
.b3 { width: 300px; height: 300px; background: #F59E0B; top: 40%; left: 60%; animation-duration: 27s; animation-delay: -4s; }

@keyframes blob-float {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(40px, 50px) scale(1.12); }
}

/* ── Grid overlay ── */
.grid-bg {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(245,158,11,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(245,158,11,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
}

/* ── Page wrapper ── */
.page-wrap {
    position: relative; z-index: 1;
    min-height: 100vh; padding: 0 24px 60px;
}

/* ── TOP NAVBAR ── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 0 28px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 40px;
}
.nav-logo { display: flex; align-items: center; gap: 12px; }
.nav-logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #F59E0B, #FBBF24);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 28px rgba(245,158,11,0.4);
}
.nav-logo-text {
    font-family: 'Fira Code', monospace;
    font-size: 22px; font-weight: 700;
    color: #F8FAFC;
}
.nav-logo-text span { color: #F59E0B; }
.nav-badge {
    padding: 6px 14px;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 99px;
    font-size: 12px; font-weight: 600; color: #F59E0B;
    letter-spacing: 0.05em;
}

/* ── HERO ── */
.hero {
    text-align: center; padding: 0 0 48px;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 16px;
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 99px;
    font-size: 12px; font-weight: 600; color: #a78bfa;
    letter-spacing: 0.08em;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 52px; font-weight: 800;
    line-height: 1.1; letter-spacing: -1px;
    margin-bottom: 16px;
}
.hero-title .grad {
    background: linear-gradient(135deg, #F59E0B, #FBBF24, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 17px; color: #94A3B8;
    max-width: 580px; margin: 0 auto 32px;
    line-height: 1.7;
}
.hero-stats {
    display: flex; justify-content: center; gap: 40px;
}
.stat { text-align: center; }
.stat-num {
    font-family: 'Fira Code', monospace;
    font-size: 26px; font-weight: 700;
    color: #F59E0B;
    display: block;
}
.stat-lbl { font-size: 12px; color: #64748B; letter-spacing: 0.05em; }

/* ── SECTION HEADERS ── */
.section-head {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px;
}
.section-num {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #F59E0B, #FBBF24);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; color: #0F172A;
}
.section-title {
    font-size: 16px; font-weight: 700; color: #F8FAFC;
    letter-spacing: 0.02em;
}
.section-sub { font-size: 12px; color: #64748B; margin-top: 2px; }

/* ── GLASS CARDS ── */
.glass-card {
    background: rgba(34,39,53,0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    transition: border-color 0.3s;
}
.glass-card:hover { border-color: rgba(245,158,11,0.15); }

/* ── INPUT OVERRIDES (Streamlit) ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] input,
div[data-baseweb="slider"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #F8FAFC !important;
    border-radius: 12px !important;
    font-family: 'Fira Sans', sans-serif !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] input:focus {
    border-color: rgba(245,158,11,0.4) !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.15) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: 'Fira Sans', sans-serif !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #F59E0B !important;
    border-color: #F59E0B !important;
}
[data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #64748B !important;
    font-size: 11px !important;
}

/* ── PREDICT BUTTON ── */
div[data-testid="stButton"] button {
    width: 100% !important;
    padding: 18px 32px !important;
    background: linear-gradient(135deg, #F59E0B, #FBBF24) !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Fira Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    cursor: pointer !important;
    box-shadow: 0 8px 28px rgba(245,158,11,0.4) !important;
    transition: all 0.3s !important;
    letter-spacing: 0.03em !important;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 36px rgba(245,158,11,0.55) !important;
}

/* ── RESULT CARDS ── */
.result-safe {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(16,185,129,0.08));
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 20px; padding: 32px;
    text-align: center;
    animation: result-in 0.5s cubic-bezier(0.16,1,0.3,1);
    box-shadow: 0 0 40px rgba(34,197,94,0.08);
}
.result-churn {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px; padding: 32px;
    text-align: center;
    animation: result-in 0.5s cubic-bezier(0.16,1,0.3,1);
    box-shadow: 0 0 40px rgba(239,68,68,0.08);
}
@keyframes result-in {
    from { opacity:0; transform: scale(0.9) translateY(20px); }
    to   { opacity:1; transform: scale(1) translateY(0); }
}
.result-emoji { font-size: 56px; margin-bottom: 12px; display: block; }
.result-verdict {
    font-size: 26px; font-weight: 800; margin-bottom: 8px;
    letter-spacing: -0.5px;
}
.result-verdict-safe  { color: #22c55e; }
.result-verdict-churn { color: #ef4444; }
.result-detail { font-size: 14px; color: #94A3B8; line-height: 1.6; }
.result-prob {
    font-family: 'Fira Code', monospace;
    font-size: 42px; font-weight: 700;
    margin: 16px 0 8px;
}
.result-prob-safe  { color: #22c55e; }
.result-prob-churn { color: #ef4444; }

/* ── RISK METER ── */
.risk-bar-wrap {
    margin: 20px 0;
    background: rgba(255,255,255,0.05);
    border-radius: 99px; height: 8px;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.8s cubic-bezier(0.16,1,0.3,1);
}

/* ── FEATURE CHIPS ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 99px;
    font-size: 12px; color: #94A3B8;
    font-family: 'Fira Code', monospace;
}

/* ── PROFILE CARD ── */
.profile-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
    margin-bottom: 16px;
}
.profile-kv {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 12px 16px;
}
.profile-k { font-size: 10px; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.profile-v { font-size: 14px; font-weight: 600; color: #F8FAFC; font-family: 'Fira Code', monospace; }

/* ── FOOTER ── */
.footer {
    text-align: center; padding: 32px 0 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 48px;
    font-size: 12px; color: #334155;
}
</style>

<!-- Animated background -->
<div class="blob-container">
    <div class="blob b1"></div>
    <div class="blob b2"></div>
    <div class="blob b3"></div>
</div>
<div class="grid-bg"></div>

<div class="page-wrap">
<!-- ── NAVBAR ── -->
<div class="navbar">
    <div class="nav-logo">
        <div class="nav-logo-icon">💠</div>
        <span class="nav-logo-text">Churn<span>AI</span></span>
    </div>
    <div class="nav-badge">⚡ ANN Prediction Engine v2.0</div>
</div>

<!-- ── HERO ── -->
<div class="hero">
    <div class="hero-tag">🤖 Neural Network Powered &nbsp;·&nbsp; Real-time Inference</div>
    <h1 class="hero-title">Customer Churn<br><span class="grad">Prediction Engine</span></h1>
    <p class="hero-sub">
        An Artificial Neural Network trained on 10,000+ customer profiles to predict churn risk with high accuracy.
        Fill in the customer details to get an instant AI prediction.
    </p>
    <div class="hero-stats">
        <div class="stat"><span class="stat-num">10K+</span><span class="stat-lbl">Training Samples</span></div>
        <div class="stat"><span class="stat-num">ANN</span><span class="stat-lbl">Model Architecture</span></div>
        <div class="stat"><span class="stat-num">3</span><span class="stat-lbl">Geographies</span></div>
        <div class="stat"><span class="stat-num">~86%</span><span class="stat-lbl">Accuracy</span></div>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# ─── LAYOUT ─────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    # ── Section 1: Customer Profile ──
    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <div class="section-num">1</div>
            <div>
                <div class="section-title">Customer Profile</div>
                <div class="section-sub">Demographics and account identity</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    with c2:
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    with c3:
        age = st.slider('🎂 Age', 18, 92, 35)

    # ── Section 2: Financial Details ──
    st.markdown("""
    <div class="glass-card" style="margin-top:20px;">
        <div class="section-head">
            <div class="section-num">2</div>
            <div>
                <div class="section-title">Financial Details</div>
                <div class="section-sub">Banking and account metrics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c4, c5 = st.columns(2)
    with c4:
        credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=900, value=650, step=1)
        balance = st.number_input('💰 Account Balance ($)', min_value=0.0, value=50000.0, step=100.0)
    with c5:
        estimated_salary = st.number_input('💵 Estimated Salary ($)', min_value=0.0, value=75000.0, step=1000.0)
        tenure = st.slider('📅 Tenure (Years)', 0, 10, 4)

    # ── Section 3: Product & Activity ──
    st.markdown("""
    <div class="glass-card" style="margin-top:20px;">
        <div class="section-head">
            <div class="section-num">3</div>
            <div>
                <div class="section-title">Product & Activity</div>
                <div class="section-sub">Engagement and product holdings</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c6, c7, c8 = st.columns(3)
    with c6:
        num_of_products = st.slider('📦 Products', 1, 4, 2)
    with c7:
        has_cr_card = st.selectbox('💳 Has Credit Card', [1, 0], format_func=lambda x: '✅ Yes' if x == 1 else '❌ No')
    with c8:
        is_active_member = st.selectbox('🏃 Active Member', [1, 0], format_func=lambda x: '✅ Yes' if x == 1 else '❌ No')

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    predict_btn = st.button('🔮  Run AI Prediction', use_container_width=True)

with col_right:
    st.markdown("""
    <div class="glass-card">
        <div class="section-head">
            <div class="section-num" style="background:linear-gradient(135deg,#8B5CF6,#a78bfa);">4</div>
            <div>
                <div class="section-title">AI Prediction Result</div>
                <div class="section-sub">Neural network output</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#334155;">
            <div style="font-size:56px; margin-bottom:16px; opacity:0.4;">🤖</div>
            <div style="font-size:15px; color:#475569;">Fill in the customer details on the left<br>and click <strong style="color:#F59E0B;">Run AI Prediction</strong> to get results.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── BACKEND: Prepare input & predict (unchanged) ──
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        # ── RENDER RESULT ──
        pct = round(prediction_proba * 100, 1)
        bar_color = "#22c55e" if prediction_proba <= 0.5 else "#ef4444"
        bar_grad = f"linear-gradient(90deg, {bar_color}, {'#4ade80' if prediction_proba <= 0.5 else '#f87171'})"

        if prediction_proba <= 0.5:
            st.markdown(f"""
            <div class="result-safe">
                <span class="result-emoji">✅</span>
                <div class="result-verdict result-verdict-safe">Customer Will Stay</div>
                <div class="result-prob result-prob-safe">{pct}%</div>
                <div style="font-size:12px;color:#64748B;margin-bottom:16px;">CHURN PROBABILITY</div>
                <div class="risk-bar-wrap">
                    <div class="risk-bar-fill" style="width:{pct}%; background:{bar_grad};"></div>
                </div>
                <div style="margin-top:16px; font-size:13px; color:#94A3B8; line-height:1.7;">
                    The model predicts this customer is <strong style="color:#22c55e;">unlikely to churn</strong>.
                    They show strong engagement signals and financial stability.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-churn">
                <span class="result-emoji">⚠️</span>
                <div class="result-verdict result-verdict-churn">High Churn Risk</div>
                <div class="result-prob result-prob-churn">{pct}%</div>
                <div style="font-size:12px;color:#64748B;margin-bottom:16px;">CHURN PROBABILITY</div>
                <div class="risk-bar-wrap">
                    <div class="risk-bar-fill" style="width:{pct}%; background:{bar_grad};"></div>
                </div>
                <div style="margin-top:16px; font-size:13px; color:#94A3B8; line-height:1.7;">
                    The model predicts this customer is <strong style="color:#ef4444;">likely to churn</strong>.
                    Consider immediate retention strategies and personalized offers.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Customer summary profile ──
        st.markdown(f"""
        <div style="margin-top:20px;">
            <div style="font-size:11px;font-weight:700;color:#64748B;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;">📋 Analyzed Profile</div>
            <div class="profile-row">
                <div class="profile-kv"><div class="profile-k">Geography</div><div class="profile-v">{geography}</div></div>
                <div class="profile-kv"><div class="profile-k">Gender</div><div class="profile-v">{gender}</div></div>
                <div class="profile-kv"><div class="profile-k">Age</div><div class="profile-v">{age} yrs</div></div>
                <div class="profile-kv"><div class="profile-k">Credit Score</div><div class="profile-v">{credit_score}</div></div>
                <div class="profile-kv"><div class="profile-k">Balance</div><div class="profile-v">${balance:,.0f}</div></div>
                <div class="profile-kv"><div class="profile-k">Tenure</div><div class="profile-v">{tenure} yrs</div></div>
            </div>
            <div class="chip-row">
                <span class="chip">📦 {num_of_products} Product{'s' if num_of_products>1 else ''}</span>
                <span class="chip">{'💳 Has Card' if has_cr_card else '🚫 No Card'}</span>
                <span class="chip">{'🏃 Active' if is_active_member else '💤 Inactive'}</span>
                <span class="chip">💵 ${estimated_salary:,.0f}/yr</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    ChurnAI &nbsp;·&nbsp; Built with TensorFlow & Streamlit &nbsp;·&nbsp; ANN Classification Model
</div>
""", unsafe_allow_html=True)
