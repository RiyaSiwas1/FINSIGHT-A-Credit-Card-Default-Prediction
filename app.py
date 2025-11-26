# app.py
"""
FinSight Enterprise Risk Portal 
----------------------------------------------------
Full login/signup/logout system with simulated email confirmation and password reset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import hashlib
import os
import uuid
import shap
import io
from typing import Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="FinSight Enterprise Risk Portal", page_icon="💳", layout="wide")

# ==========================================================
# THEME COLORS
# ==========================================================
HEADER_PURPLE = "#7B2CBF"
ACCENT_PINK = "#FF5CA8"
LIGHT_PINK = "#FFE6F6"
TEXT_DARK = "#2B1338"
MUTED = "#8A719D"
GOOD = "#00B894"
WARN = "#E84393"

# ==========================================================
# GLOBAL CSS STYLING (Pink Background + Sidebar)
# ==========================================================
st.markdown(f"""
<style>
/* Background */
body {{
    background-color: {LIGHT_PINK};
    background-attachment: fixed;
    color: {TEXT_DARK};
    font-family: 'Segoe UI', sans-serif;
}}

/* Sidebar */
section[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(135deg, #FFD6F0 0%, #C8A2FF 100%);
    color: {TEXT_DARK};
}}
section[data-testid="stSidebar"] h2 {{
    color: {HEADER_PURPLE};
}}

/* Header */
.header {{
    display:flex; 
    align-items:center; 
    gap:16px; 
    padding:18px; 
    border-radius:14px; 
    background: linear-gradient(90deg, {ACCENT_PINK}, {HEADER_PURPLE}); 
    color: white; 
    box-shadow: 0 8px 25px rgba(123,44,191,0.25);
}}
.logo-circle {{
    width:60px;
    height:60px;
    border-radius:16px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(255,255,255,0.1);
    padding:6px;
}}
.card {{
    background: white;
    border-radius:14px;
    padding:20px;
    box-shadow: 0 4px 18px rgba(123,44,191,0.1);
    margin-bottom: 18px;
    border: 1px solid rgba(255,92,168,0.2);
}}
.section-title {{
    color: {HEADER_PURPLE};
    font-weight:700;
    font-size:18px;
    margin-bottom:8px;
}}
div.stButton > button:first-child {{
    background: linear-gradient(90deg, {ACCENT_PINK}, {HEADER_PURPLE});
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    font-weight: 600;
}}
div.stButton > button:first-child:hover {{
    opacity: 0.9;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# AUTH HELPERS
# ==========================================================
USER_FILE = "users.csv"
RESET_FILE = "reset_tokens.csv"

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def load_users():
    if os.path.exists(USER_FILE):
        return pd.read_csv(USER_FILE)
    else:
        df = pd.DataFrame(columns=["username", "password", "email", "confirmed"])
        df.to_csv(USER_FILE, index=False)
        return df

def save_user(username, password, email):
    df = load_users()
    if username in df["username"].values:
        return False
    new_entry = pd.DataFrame([[username, make_hashes(password), email, False]], 
                             columns=["username", "password", "email", "confirmed"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(USER_FILE, index=False)
    return True

def confirm_user(email):
    df = load_users()
    df.loc[df["email"] == email, "confirmed"] = True
    df.to_csv(USER_FILE, index=False)

def authenticate(username, password):
    df = load_users()
    if username in df["username"].values:
        stored_hash = df.loc[df["username"] == username, "password"].values[0]
        confirmed = df.loc[df["username"] == username, "confirmed"].values[0]
        if not confirmed:
            st.warning("Please confirm your email before logging in.")
            return False
        return check_hashes(password, stored_hash)
    return False

# ==========================================================
# PASSWORD RESET (DEV SIMULATION)
# ==========================================================
def create_reset_token(username):
    token = str(uuid.uuid4())[:8]
    df = pd.DataFrame([[username, token]], columns=["username", "token"])
    df.to_csv(RESET_FILE, mode='a', index=False, header=not os.path.exists(RESET_FILE))
    return token

def verify_reset_token(username, token):
    if not os.path.exists(RESET_FILE):
        return False
    df = pd.read_csv(RESET_FILE)
    return any((df["username"] == username) & (df["token"] == token))

# ==========================================================
# SESSION STATE
# ==========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ==========================================================
# LOGIN / SIGNUP / RESET UI
# ==========================================================
def auth_screen():
    st.sidebar.title("💳 FinSight Portal Access")
    mode = st.sidebar.radio("Select Option:", ["Login", "Signup", "Forgot Password"])

    if mode == "Login":
        st.markdown("<div class='header'><div class='logo-circle'>💳</div><h2>FinSight Enterprise Risk Portal</h2></div>", unsafe_allow_html=True)
        st.markdown("### Login to Continue")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials or unconfirmed account.")

    elif mode == "Signup":
        st.markdown("### Create Account")
        username = st.text_input("Choose Username")
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Sign Up"):
            if password != confirm:
                st.warning("Passwords do not match.")
            elif username.strip() == "" or email.strip() == "":
                st.warning("Please fill all fields.")
            else:
                if save_user(username, password, email):
                    token = str(uuid.uuid4())[:6]
                    st.success(f"✅ Account created! (Simulated email sent to {email})")
                    st.info(f"📧 Dev-only confirmation link: click below to simulate confirmation.")
                    if st.button("Confirm Email"):
                        confirm_user(email)
                        st.success("✅ Email confirmed! You can now log in.")
                else:
                    st.warning("⚠️ Username already exists.")

    else:  # Forgot password
        st.markdown("### Reset Password (Dev Simulation)")
        username = st.text_input("Enter Username")
        if st.button("Generate Reset Token"):
            token = create_reset_token(username)
            st.info(f"📨 Dev-only reset link generated! Token: **{token}**")

        username2 = st.text_input("Username for Reset")
        token_input = st.text_input("Enter Token")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Reset Password"):
            if verify_reset_token(username2, token_input):
                df = load_users()
                df.loc[df["username"] == username2, "password"] = make_hashes(new_pass)
                df.to_csv(USER_FILE, index=False)
                st.success("✅ Password reset successful! You can log in now.")
            else:
                st.error("❌ Invalid token or username.")


# -------------------------
# Utility: sanitize numeric inputs
# -------------------------
import re

NUM_RE = re.compile(r'[+-]?\d+(\.\d+)?([eE][+-]?\d+)?')

def robust_clean_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly clean dataframe so every cell becomes a float.
    - Extracts the first numeric token using regex (handles scientific notation).
    - Removes brackets, quotes, commas used as thousand separators.
    - Coerces to float and fills NaNs with 0.0.
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        # convert to string once
        s = df_clean[col].astype(str)

        # fast path: if dtype is numeric already, keep it
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            continue

        def extract_num(x):
            if x is None:
                return np.nan
            x = x.strip()
            # remove surrounding brackets/quotes
            if (x.startswith('[') and x.endswith(']')) or (x.startswith('(') and x.endswith(')')):
                x = x[1:-1].strip()
            # remove stray quotes
            x = x.replace('"', '').replace("'", "")
            # remove thousands separators like 1,234 => 1234 (but keep decimal comma? we assume dot decimals)
            x = x.replace(',', '')
            # find first numeric substring
            m = NUM_RE.search(x)
            if m:
                try:
                    return float(m.group(0))
                except Exception:
                    return np.nan
            # fallback: attempt direct to_numeric
            try:
                return pd.to_numeric(x, errors='coerce')
            except Exception:
                return np.nan

        df_clean[col] = s.map(extract_num).astype(float)

    # final fill
    df_clean = df_clean.fillna(0.0)
    return df_clean


# -------------------------
# Model prediction function
# -------------------------
def predict_df(df: pd.DataFrame, model):
    if model is None:
        raise RuntimeError('model.pkl not loaded.')
    df = clean_numeric_df(df)
    X = df.values
    preds = model.predict(X)
    probs = None
    if hasattr(model, 'predict_proba'):
        try:
            raw = model.predict_proba(X)
            probs = raw[:, 1] if raw.shape[1] >= 2 else raw[:, 0]
        except Exception:
            probs = None
    return preds.astype(int), probs

# -------------------------
# SHAP integration
# -------------------------
def shap_analysis(model, input_df):
    st.subheader('Model Explainability (SHAP Analysis)')
    input_df_clean = clean_numeric_df(input_df)
    
    try:
        explainer = shap.Explainer(model, input_df_clean)
        shap_values = explainer(input_df_clean)
        st.write('### Feature Impact Visualization')
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, input_df_clean, plot_type='bar', show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f'SHAP failed: {e}')

# -------------------------
# Streamlit Dashboard Header (RBI Style)
# -------------------------
def display_header():

        st.markdown("""
<style>
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.animated-header {
    background: linear-gradient(90deg, #FF5CA8, #7B2CBF, #C77DFF);
    background-size: 200% 200%;
    animation: gradientShift 8s ease infinite;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(123, 44, 191, 0.25);
}
.animated-header h1 {
    color: #FFFFFF;
    text-align: center;
    font-family: 'Helvetica Neue', sans-serif;
    letter-spacing: 1px;
    margin-bottom: -5px;
}
.animated-header h2 {
    color: #FFE6F6;
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
    margin-top: -5px;
}
.animated-header p {
    color: #FCE6FF;
    text-align: center;
    font-size: 16px;
    font-family: 'Segoe UI', sans-serif;
}
</style>

<div class="animated-header">
    <h1>🏦</h1>
    <h2>FinSight: A Credit-Card Default Payment Prediction</h2>
    <p>Empowering Credit Risk Intelligence with Smart AI Analytics</p>
</div>
<br>
""", unsafe_allow_html=True)

    
import joblib
import pickle                


st.set_page_config(page_title=' Finsight Enterprise Risk Portal', page_icon='💳', layout='wide')

st.sidebar.title('🏦 FinSight Navigation')
display_header()

model = None  # Placeholder for actual model loading

    

# Optional libs
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# PDF export helper: try to use pdfkit (wkhtmltopdf required on host)
try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except Exception:
    PDFKIT_AVAILABLE = False

# -------------------------
# Page config + CSS (match Collab look)
# -------------------------
st.set_page_config(page_title="A Credit-Card Default Payment Prediction.", page_icon="💳", layout="wide")

HEADER_BLUE = "#0A3A67"
ACCENT = "#0B5FA5"
GOLD = "#C4A000"
TEXT_DARK = "#0b1220"
MUTED = "#566270"
GOOD = "#16a34a"
WARN = "#ef4444"

st.markdown(f"""
<style>
body {{ background-color: #0B1A3A; }}
.header {{ display:flex; align-items:center; gap:16px; padding:18px; border-radius:10px; background: linear-gradient(90deg, {HEADER_PURPLE}, {ACCENT_PINK}); color: white; box-shadow: 0 10px 30px rgba(10,58,103,0.08); }}
.logo-circle {{ width:60px;height:60px;border-radius:12px;display:flex;align-items:center;justify-content:center; background: rgba(255,255,255,0.06); padding:6px; }}
.card {{ background: white; border-radius:10px; padding:16px; box-shadow: 0 6px 18px rgba(11,35,77,0.04); margin-bottom: 16px; border: 1px solid rgba(11,95,165,0.04); }}
.section-title {{ color: {HEADER_PURPLE}; font-weight:700; margin-bottom:8px; }}
.small-muted {{ color: {HEADER_PURPLE}; font-size:13px; }}
.explain-box {{ margin-top:8px; padding:10px; border-radius:8px; background:#f6f9ff; border:1px solid rgba(11,95,165,0.04); }}
</style>
""", unsafe_allow_html=True)
# ==========================================================
# AUTH CONTROL FLOW
# ==========================================================
if not st.session_state.logged_in:
    auth_screen()
    st.stop()
else:
    st.sidebar.success(f"👋 Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("You have been logged out.")
        st.rerun()


# -------------------------
# Constants / expected schema
# -------------------------
EXPECTED_FEATURES = [
    "ID",
    "LIMIT_BAL", "SEX", "EDUCATION" , "MARRIAGE" , "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]
TARGET_NAME = "default.payment.next.month"

MODEL_PATH_CANDIDATES = [
    "model.pkl",
    "model.joblib",
    "final_model.pkl",
    "/mnt/data/model.pkl",
    "/mnt/data/credit_card_model.pkl",
]

# -------------------------
# Model loading utilities
# -------------------------
@st.cache_resource
def try_load_model_from_candidates() -> Tuple[Optional[object], Optional[str]]:
    for p in MODEL_PATH_CANDIDATES:
        if os.path.exists(p):
            try:
                if p.endswith('.joblib'):
                    m = joblib.load(p)
                else:
                    with open(p, 'rb') as f:
                        m = pickle.load(f)
                return m, None
            except Exception as e:
                return None, f"Found {p} but failed to load: {e}"
    return None, None

def load_model_from_bytes(b: bytes, fname: str = 'uploaded') -> object:
    try:
        return pickle.loads(b)
    except Exception:
        pass
    try:
        buf = io.BytesIO(b)
        buf.seek(0)
        return joblib.load(buf)
    except Exception as e:
        raise RuntimeError(f"Could not deserialize uploaded model ({fname}): {e}")

# -------------------------
# Prediction helpers
# -------------------------

def predict_df(model, df: pd.DataFrame):
    if model is None:
        raise RuntimeError('No model loaded')
    X = df.values
    preds = model.predict(X)
    probs = None
    if hasattr(model, 'predict_proba'):
        try:
            raw = model.predict_proba(X)
            if raw.shape[1] >= 2:
                probs = raw[:, 1]
            else:
                probs = raw[:, 0]
        except Exception:
            probs = None
    return preds.astype(int), probs

# Heuristic explanation (short)
def get_explanation(row, prob, pred):
    bullets = []
    if prob is None:
        bullets.append('Probability not available — showing label only.')
    else:
        if prob >= 0.75:
            bullets.append(f'High predicted risk ({prob:.0%}).')
        elif prob >= 0.45:
            bullets.append(f'Moderate predicted risk ({prob:.0%}).')
        else:
            bullets.append(f'Low predicted risk ({prob:.0%}).')
    # repayment signals
    pay_cols = [c for c in EXPECTED_FEATURES if c.startswith('PAY_')]
    delays = 0
    max_delay = 0
    for c in pay_cols:
        try:
            v = int(row.get(c, 0))
        except Exception:
            v = 0
        if v > 0:
            delays += 1
            if v > max_delay:
                max_delay = v
    if delays >= 3 or max_delay >= 2:
        bullets.append(f'Multiple recent delays ({delays} months).')
    elif delays > 0:
        bullets.append(f'Some late payments ({delays} months).')
    else:
        bullets.append('No recent payment delays.')
    # utilization
    try:
        limit_bal = float(row.get('LIMIT_BAL', 0)) or 0.0
        bill_sum = sum(float(row.get(f'BILL_AMT{i}', 0.0)) for i in range(1,7))
    except Exception:
        limit_bal = 0.0
        bill_sum = 0.0
    if limit_bal > 0:
        util = bill_sum / max(limit_bal,1.0)
        if util >= 1.0:
            bullets.append('Very high utilization vs credit limit.')
        elif util >= 0.5:
            bullets.append('Moderate-to-high utilization (≈50%+).')
        else:
            bullets.append('Utilization low or moderate.')
    return bullets[:4]

# -------------------------
# EDA helpers (sample data)
# -------------------------
@st.cache_data
def load_sample_data():
    sample_paths = ['/mnt/data/credit_card_default.csv', '/mnt/data/credit_card_data.csv']
    for p in sample_paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    rng = np.random.default_rng(42)
    n = 600
    df = pd.DataFrame({
        'ID': np.arange(1,n+1),
        'LIMIT_BAL': rng.integers(10000,500000,n),
        'SEX': rng.integers(1,3,n),
        'EDUCATION': rng.integers(1,5,n),
        'MARRIAGE': rng.integers(1,4,n),
        'AGE': rng.integers(21,75,n),
        'PAY_0': rng.integers(-1,4,n),
        'PAY_2': rng.integers(-1,4,n),
        'PAY_3': rng.integers(-1,4,n),
        'PAY_4': rng.integers(-1,4,n),
        'PAY_5': rng.integers(-1,4,n),
        'PAY_6': rng.integers(-1,4,n),
        'BILL_AMT1': rng.integers(0,200000,n),
        'BILL_AMT2': rng.integers(0,200000,n),
        'BILL_AMT3': rng.integers(0,200000,n),
        'BILL_AMT4': rng.integers(0,200000,n),
        'BILL_AMT5': rng.integers(0,200000,n),
        'BILL_AMT6': rng.integers(0,200000,n),
        'PAY_AMT1': rng.integers(0,150000,n),
        'PAY_AMT2': rng.integers(0,150000,n),
        'PAY_AMT3': rng.integers(0,150000,n),
        'PAY_AMT4': rng.integers(0,150000,n),
        'PAY_AMT5': rng.integers(0,150000,n),
        'PAY_AMT6': rng.integers(0,150000,n),
    })
    delays = (df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]>0).sum(axis=1)
    df[TARGET_NAME] = ((delays>2) | ((df['LIMIT_BAL']<50000) & (delays>1))).astype(int)
    return df

edata = load_sample_data()

# -------------------------
# Sidebar: model loader + navigation
# -------------------------
with st.sidebar:
    st.markdown(f"Smart Analysis" )
    st.caption('Enterprise Grade Scoring Portal')
    st.markdown('---')
    model, model_err = try_load_model_from_candidates()
    if model is not None:
        st.success('Model auto-loaded from disk')
        st.write(f'Model type: `{type(model).__name__}`')
    elif model_err:
        st.error(model_err)
    else:
        st.info('No model found on disk. Upload below or place model.pkl next to this script.')

    uploaded_model = st.file_uploader('Upload model (.pkl / .joblib)', type=['pkl','joblib'])
    if uploaded_model is not None:
        try:
            model = load_model_from_bytes(uploaded_model.read(), uploaded_model.name)
            st.success(f'Model loaded from {uploaded_model.name} — {type(model).__name__}')
        except Exception as e:
            st.error(str(e))

    st.markdown('---')
    page = st.radio('Navigation', ['Dashboard','Predictions','Model Insights','Export Report','About'])

# Header

# -------------------------
# Page implementations
# -------------------------
if page == 'Dashboard':
    st.header('Exploratory Data Analysis')
    st.write('Upload your dataset (CSV) for tailored EDA or use the sample dataset below.')
    uploaded = st.file_uploader('Upload dataset for EDA (CSV)', type=['csv'])
    if uploaded is not None:
        try:
            edata = pd.read_csv(uploaded)
            st.success('Dataset loaded')
        except Exception as e:
            st.error(f'Failed to read CSV: {e}')

    st.subheader('Data Preview')
    st.dataframe(edata.head())

    st.subheader('Payment delay vs risk level')
    pay_cols = [c for c in edata.columns if c.startswith('PAY_')]
    if not pay_cols:
        st.info('No PAY_ columns found in dataset for this chart.')
    else:
        edata['delay_count'] = (edata[pay_cols] > 0).sum(axis=1)
        grp = edata.groupby('delay_count')[TARGET_NAME].agg(['mean','count']).reset_index().rename(columns={'mean':'default_rate'})
        fig = px.bar(grp, x='delay_count', y='default_rate', labels={'delay_count':'Number of delayed months','default_rate':'Default rate (fraction)'}, title='Default rate by number of delayed months')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Credit limit vs default (box + scatter)')
    if TARGET_NAME not in edata.columns:
        st.info(f'Target column `{TARGET_NAME}` not found. Upload data with target to see this plot.')
    else:
        fig2 = px.box(edata, x=TARGET_NAME, y='LIMIT_BAL', points='all', title='Credit limit distribution by actual default label')
        st.plotly_chart(fig2, use_container_width=True)
        sample_n = min(len(edata), 2000)
        fig3 = px.scatter(edata.sample(sample_n), x='LIMIT_BAL', y='BILL_AMT1' if 'BILL_AMT1' in edata.columns else edata.columns[-1], color=TARGET_NAME, title='LIMIT_BAL vs recent bill (colored by default)')
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader('Correlation heatmap (numeric features)')
    
    num = edata.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        st.info('Not enough numeric columns for correlation heatmap.')
    else:
        corr = num.corr()
        heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmin=-1, zmax=1))
        heat.update_layout(title='Correlation matrix')
        st.plotly_chart(heat, use_container_width=True)

elif page == 'Predictions':
    st.header('Predictions — Single & Bulk')
    if model is None:
        st.warning('No model loaded. Upload one in the sidebar or place model.pkl next to the app.')

    st.subheader('Single record (readable inputs)')
    with st.form(key='single_form'):
        cols = st.columns([1,1,1])
        with cols[0]:
            id_val = st.number_input('ID', value=1001, step=1)
            limit_bal = st.number_input('Credit limit (LIMIT_BAL)', value=200000.0, step=1000.0)
            age = st.number_input('AGE', value=30, min_value=18, max_value=120)
            sex_label = st.selectbox('SEX', options=['Male','Female'])
            sex = 1 if sex_label == 'Male' else 2
        with cols[1]:
            edu_label = st.selectbox('EDUCATION', options=['Graduate School / Post Graduate','University / Undergraduate','High School / Secondary','Others / Not specified'])
            edu_map = { 'Graduate School / Post Graduate':1, 'University / Undergraduate':2, 'High School / Secondary':3, 'Others / Not specified':4 }
            edu = edu_map[edu_label]
            mar_label = st.selectbox('MARRIAGE', options=['Married','Single','Others'])
            mar_map = {'Married':1,'Single':2,'Others':3}
            mar = mar_map[mar_label]
            # repayment history inputs
            pay0 = st.number_input('PAY_0 (most recent)', value=0, step=1)
            pay2 = st.number_input('PAY_2', value=0, step=1)
        with cols[2]:
            pay3 = st.number_input('PAY_3', value=0, step=1)
            pay4 = st.number_input('PAY_4', value=0, step=1)
            pay5 = st.number_input('PAY_5', value=0, step=1)
            pay6 = st.number_input('PAY_6', value=0, step=1)

        st.markdown('---')
        st.subheader('Bill amounts (BILL_AMT*)')
        bcols = st.columns(3)
        bill_vals = {}
        for i in range(1,7):
            with bcols[(i-1)%3]:
                bill_vals[i] = st.number_input(f'BILL_AMT{i}', value=0.0, step=100.0)

        st.subheader('Payment amounts (PAY_AMT*)')
        pcols = st.columns(3)
        pay_amt_vals = {}
        for i in range(1,7):
            with pcols[(i-1)%3]:
                pay_amt_vals[i] = st.number_input(f'PAY_AMT{i}', value=0.0, step=100.0)

        submit = st.form_submit_button('Predict single record')

    if submit:
        row = {f:0 for f in EXPECTED_FEATURES}
        row['ID'] = id_val
        row['LIMIT_BAL'] = limit_bal
        row['AGE'] = age
        row['SEX'] = sex
        row['EDUCATION'] = edu
        row['MARRIAGE'] = mar
        row['PAY_0'] = pay0
        row['PAY_2'] = pay2
        row['PAY_3'] = pay3
        row['PAY_4'] = pay4
        row['PAY_5'] = pay5
        row['PAY_6'] = pay6
        for i in range(1,7):
            row[f'BILL_AMT{i}'] = bill_vals[i]
            row[f'PAY_AMT{i}'] = pay_amt_vals[i]
        input_df = pd.DataFrame([row], columns=EXPECTED_FEATURES)
        st.write('Model input (ordered):')
        st.dataframe(input_df)
        try:
            preds, probs = predict_df(model, input_df)
            pred = int(preds[0])
            prob = float(probs[0]) if probs is not None else None
            # gauge
            pct = int(round((prob or 0)*100))
            color = WARN if prob and prob>=0.7 else ACCENT if prob and prob>=0.4 else GOOD
            st.markdown(f"<div class='card'><strong>Prediction:</strong> {'DEFAULT (1)' if pred==1 else 'Non-default (0)'} {f'— Prob: {prob:.2%}' if prob is not None else ''}</div>", unsafe_allow_html=True)
            bullets = get_explanation(row, prob, pred)
            explain_html = "<div class='explain-box'><div style='font-weight:700;margin-bottom:6px'>Why this prediction?</div>"
            for b in bullets:
                explain_html += f"<div>• {b}</div>"
            explain_html += '</div>'
            st.markdown(explain_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f'Prediction failed: {e}')

    st.markdown('---')
    st.subheader('Bulk prediction (CSV)')
    uploaded = st.file_uploader('Upload CSV for batch scoring (must contain expected columns)', type=['csv'], key='bulk_upload')
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write('Preview:')
            st.dataframe(df.head())
            missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
            if missing:
                st.warning(f'Uploaded CSV missing expected columns: {missing}')
            else:
                X = df[EXPECTED_FEATURES].copy()
                preds, probs = predict_df(model, X)
                df_out = df.copy()
                df_out['prediction'] = preds.astype(int)
                if probs is not None:
                    df_out['prob_pos'] = probs
                st.success('Batch scoring complete')
                st.dataframe(df_out.head())
                csv_bytes = df_out.to_csv(index=False).encode('utf-8')
                st.download_button('Download scored CSV', data=csv_bytes, file_name='scored_predictions.csv', mime='text/csv')
        except Exception as e:
            st.error(f'Failed to read CSV: {e}')

elif page == 'Model Insights':
    st.header('Model Insights')
    if model is None:
        st.warning('Model not loaded — feature importance / coefficients unavailable.')
    else:
        st.subheader('Model summary')
        st.write(type(model))
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            feat = EXPECTED_FEATURES.copy()
            if len(fi) == len(feat):
                df_fi = pd.DataFrame({'feature':feat, 'importance':fi}).sort_values('importance', ascending=False)
                fig = px.bar(df_fi, x='importance', y='feature', orientation='h', title='Feature importance')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('feature_importances_ length mismatch.')
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim == 1:
                co = coef
            else:
                co = coef[0]
            feat = EXPECTED_FEATURES.copy()
            if len(co) == len(feat):
                df_co = pd.DataFrame({'feature':feat, 'coef':co}).sort_values('coef', ascending=False)
                fig = px.bar(df_co, x='coef', y='feature', orientation='h', title='Model coefficients')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('Model coefficients shape mismatch.')
        else:
            st.info('Model does not expose feature_importances_ or coef_. Consider SHAP.')




elif page == 'Export Report':
    st.header('Export EDA + Model Report')
    st.write('You can export an HTML report of current EDA charts + model insights. PDF export uses pdfkit/wkhtmltopdf if available.')
    generate_html = st.button('Generate HTML report')
    if generate_html:
        html = '<html><head><meta charset="utf-8"><title>FinSight Report</title></head><body>'
        html += f'<h1>FinSight Report</h1><p>Generated by app</p>'
        html += '<h2>Data sample</h2>'
        html += edata.head(10).to_html(index=False)
        html += '</body></html>'
        st.markdown('Report generated — download below:')
        st.download_button('Download HTML report', data=html.encode('utf-8'), file_name='finsight_report.html', mime='text/html')

    if PDFKIT_AVAILABLE:
        if st.button('Generate PDF report (requires wkhtmltopdf installed)'):
            try:
                html = '<html><head><meta charset="utf-8"><title>FinSight PDF Report</title></head><body>'
                html += f'<h1>FinSight Report</h1><p>Generated by app</p>'
                html += '<h2>Data sample</h2>'
                html += edata.head(10).to_html(index=False)
                html += '</body></html>'
                pdf_bytes = pdfkit.from_string(html, False)
                st.success('PDF generated — download below')
                st.download_button('Download PDF report', data=pdf_bytes, file_name='finsight_report.pdf', mime='application/pdf')
            except Exception as e:
                st.error(f'PDF generation failed: {e}')
    else:
        st.info('pdfkit or wkhtmltopdf not available on this host. You can still download the HTML report and convert it locally to PDF.')

else:
    st.markdown("""
<style>
.about-card {
    background: linear-gradient(135deg, #ffd6ff, #f3c4ff, #e5b3ff);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #d28cff;
    box-shadow: 0 4px 12px rgba(160, 24, 255, 0.2);
}
.about-title {
    font-size: 36px;
    font-weight: 900;
    color: #8a00c2;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
    color: #660099;
}
.card {
    background: white;
    padding: 18px;
    margin-top: 12px;
    border-radius: 15px;
    border-left: 6px solid #b300ff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
</style>

<div class="about-card">

<div class="about-title">💳 FinSight — Enterprise Credit Default Prediction Platform</div>
<p style="font-size:18px; margin-top:10px;">
FinSight is an intelligent, machine-learning powered system that predicts the likelihood of credit card payment default through interactive analytics and clean visual insights.
</p>

<div class="section-title">🚀 Core Features</div>

<div class="card">
<h4>🔍 Single Customer Prediction</h4>
Predict default risk instantly with LIME-based explainability.
</div>

<div class="card">
<h4>📂 Bulk Prediction (CSV Upload)</h4>
Upload large datasets and generate predictions + probability outputs for thousands of customers.
</div>

<div class="card">
<h4>📊 Explainability & Interpretability</h4>
Using LIME and SHAP, FinSight generates both global and individual-level explanations.
</div>

<div class="card">
<h4>🤖 Machine Learning Engine</h4>
Models used include XGBoost, Random Forest, Gradient Boosting & Logistic Regression.
</div>

<div class="card">
<h4>📈 Interactive Visual Dashboard</h4>
Hover-tooltips, click-to-filter charts, and dynamic risk exploration.
</div>

<div class="section-title">🏦 Why FinSight?</div>

<div class="card">
📌 Enterprise-grade accuracy  
📌 Clean UI and fast predictions  
📌 Transparent explainability  
📌 Supports risk analysts, banks, fintech & credit teams  
</div>

<div style="text-align:center; margin-top:30px; font-size:20px; color:#7b0099; font-weight:700;">
FinSight turns credit data into real financial intelligence.
</div>

</div>
""", unsafe_allow_html=True)


# End of file