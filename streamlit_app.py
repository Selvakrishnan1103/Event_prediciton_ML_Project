import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sqlite3
import hashlib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

MODEL_PATH = "attendance_model.pkl"
SCALER_PATH = "scaler.pkl"
DB_PATH = "app.db"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db_if_missing():
    
    if not os.path.exists(DB_PATH):
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                age INTEGER,
                gender INTEGER,
                reg_time INTEGER,
                distance REAL,
                event_type INTEGER,
                past_att INTEGER,
                reminder INTEGER,
                ticket INTEGER,
                weekend INTEGER,
                predicted INTEGER,
                probability REAL
            )
        ''')
      
        admin_user = "admin"
        admin_pass = "admin123"
        cur.execute("SELECT * FROM users WHERE username=?", (admin_user,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (admin_user, hash_password(admin_pass), "admin")
            )
            print("Created default admin (change password after first login)")
        conn.commit()
        conn.close()

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model = None
    scaler = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(f"Unable to load model from {MODEL_PATH}: {e}")
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            st.warning(f"Unable to load scaler from {SCALER_PATH}: {e}")
    return model, scaler

model, scaler = load_model_and_scaler()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

def login_user(username, password):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT password_hash, role FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if row:
        stored_hash, role = row
        if hash_password(password) == stored_hash:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = role
            return True, "Login successful"
    return False, "Invalid username or password"

def logout_user():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

def register_user(username, password, role="user"):
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, hash_password(password), role))
        conn.commit()
        conn.close()
        return True, "User registered"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists"

def make_prediction(input_array):
    """
    input_array: shape (1, 9) matching features:
    [age, gender, reg_time, distance, event_type, past_att, reminder, ticket, weekend]
    """
    global model, scaler
    if model is None:
        raise ValueError("Model is not loaded. Ask admin to upload a model.")
    try:
        if scaler is not None:
            input_scaled = scaler.transform(input_array)
        else:
            input_scaled = input_array
    except Exception:
        input_scaled = input_array  

    try:
        prob = model.predict_proba(input_array)[0][1]
    except Exception:
        pred = model.predict(input_array)[0]
        prob = float(pred)  
        return int(pred), prob

    pred = int(model.predict(input_array)[0])
    return pred, float(prob)

def log_prediction(username, features, pred, prob):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO predictions 
            (username, age, gender, reg_time, distance, event_type, past_att, reminder, ticket, weekend, predicted, probability)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        username,
        int(features[0]), int(features[1]), int(features[2]), float(features[3]), int(features[4]),
        int(features[5]), int(features[6]), int(features[7]), int(features[8]),
        int(pred), float(prob)
    ))
    conn.commit()
    conn.close()

init_db_if_missing()

st.set_page_config(page_title="Event Attendance — Pro", layout="wide")

st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.main .block-container {
    padding: 2rem 2rem 2rem 2rem;
}

div[data-testid="stMarkdownContainer"] h1 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 700;
}
div[data-testid="stMarkdownContainer"] h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}

.stCard, .stFrame {
    background-color: #ffffff !important;
    padding: 20px !important;
    border-radius: 15px !important;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08) !important;
    margin-bottom: 20px !important;
}

.stButton>button {
    background-color: #00aaff !important;
    color: #fff !important;
    font-weight: 600;
    border-radius: 10px !important;
    padding: 0.6em 1.2em !important;
    transition: background 0.3s ease;
}
.stButton>button:hover {
    background-color: #008ecc !important;
}

[data-testid="stSidebar"] .css-1d391kg {
    background-color: #f0f4f8 !important;
}
[data-testid="stSidebar"] h2, h3, h4, h5 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}

.stDataFrameContainer {
    border-radius: 12px !important;
    box-shadow: 0px 2px 12px rgba(0,0,0,0.05) !important;
}

a {
    color: #0077cc;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}

[data-testid="metric-container"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}

.stMarkdown h1 {
    background: linear-gradient(to right, #00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")

if not st.session_state.logged_in:
    st.sidebar.info("🔐 Please login to access your dashboard")
    allowed_pages = ["Home", "Login", "Register", "About"]
else:
    if st.session_state.role == "admin":
        allowed_pages = ["Home", "Single Prediction", "Bulk Prediction", "Dashboard (Analytics)", "Admin Panel", "About"]
    else:
        allowed_pages = ["Home", "Single Prediction", "Bulk Prediction", "Dashboard (Analytics)", "About"]

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

page = st.sidebar.selectbox("Menu", allowed_pages, index=allowed_pages.index(st.session_state.current_page) if st.session_state.current_page in allowed_pages else 0)

st.session_state.current_page = page


st.sidebar.markdown("---")
if st.session_state.logged_in:
    st.sidebar.write(f"👤 Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        logout_user()
        st.session_state.current_page = "Home" 
        st.session_state["rerun_flag"] = not st.session_state.get("rerun_flag", False)
        st.experimental_set_query_params(dummy=datetime.now().timestamp())

if page == "Home":
    st.subheader("Welcome")
    st.write("""
    Use the sidebar to navigate.  
    ➤ Login to save prediction history.  
    ➤ Admin can upload new model files and view prediction logs.
    """)
    st.info("Pro features: login, admin model upload, prediction logs, CSV bulk upload, analytics.")
    st.write("Quick actions:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model loaded", "Yes" if model is not None else "No")
    with col2:
        st.metric("Scaler loaded", "Yes" if scaler is not None else "No")
    with col3:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions")
        total_preds = cur.fetchone()[0]
        conn.close()
        st.metric("Predictions logged", total_preds)

elif page == "Login":
    st.subheader("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            ok, msg = login_user(username.strip(), password)
            if ok:
                st.success(msg)
                st.session_state.current_page = "Home"
                st.session_state["rerun_flag"] = not st.session_state.get("rerun_flag", False)
                st.experimental_set_query_params(dummy=datetime.now().timestamp())
            else:
                st.error(msg)


elif page == "Register":
    st.subheader("Create account")
    with st.form("reg_form", clear_on_submit=True):
        new_user = st.text_input("Choose username")
        new_pass = st.text_input("Choose password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Register")
        if submitted:
            if new_pass != confirm:
                st.error("Passwords do not match")
            else:
                ok, m = register_user(new_user.strip(), new_pass)
                if ok:
                    st.success("Registration successful. You are now logged in.")
                    st.session_state.logged_in = True
                    st.session_state.username = new_user.strip()
                    st.session_state.role = "user"
                    st.session_state.current_page = "Home"  # Redirect to Home
                    st.session_state["rerun_flag"] = not st.session_state.get("rerun_flag", False)
                    st.experimental_set_query_params(dummy=datetime.now().timestamp())
                else:
                    st.error(m)


elif page == "Single Prediction":
    st.subheader("🧍 Single Prediction")
    if model is None:
        st.error("Model not loaded. Ask admin to upload a model file (.pkl).")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_val = 1 if gender == "Male" else 0
        reg_time = st.number_input("Registered Days Before Event", 0, 365, 10)
        distance = st.number_input("Distance (km)", 0.0, 500.0, 10.0, step=0.1)
    with col2:
        event_map = {"Workshop": 0, "Seminar": 1, "Tech Talk": 2, "Cultural": 3}
        event_type = st.selectbox("Event Type", list(event_map.keys()))
        event_type_val = event_map[event_type]
        past_att = st.number_input("Past Attendance Count", 0, 100, 2)
        reminder = st.selectbox("Reminder Sent?", ["No", "Yes"])
        reminder_val = 1 if reminder == "Yes" else 0
        ticket = st.number_input("Ticket Price", 0, 10000, 500)
        weekend = st.selectbox("Weekend Event?", ["No", "Yes"])
        weekend_val = 1 if weekend == "Yes" else 0

    features = [age, gender_val, reg_time, distance, event_type_val, past_att, reminder_val, ticket, weekend_val]
    if st.button("Predict"):
        with st.spinner("Running model..."):
            try:
                x = np.array([features], dtype=float)
                pred, prob = make_prediction(x)
                prob_pct = prob * 100 if prob <= 1 else prob
                if pred == 1:
                    st.success(f"✔ Likely to attend — Probability: {prob_pct:.2f}%")
                else:
                    st.error(f"✘ Unlikely to attend — Probability: {prob_pct:.2f}%")

                user = st.session_state.username if st.session_state.logged_in else "anonymous"
                log_prediction(user, features, pred, prob_pct)
                st.balloons()
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "Bulk Prediction":
    st.subheader("📁 Bulk prediction (CSV)")
    st.markdown("""
    Upload a CSV with the following columns (no header row reorder needed):
    `age, gender, registration_time_days_before, location_distance_km, event_type, past_attendance_count, reminder_sent, ticket_price, is_weekend_event`
    **Note:** categorical values must be numeric consistent with mapping used in app:
    gender: Male=1, Female=0
    event_type mapping: Workshop=0, Seminar=1, Tech Talk=2, Cultural=3
    """)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.markdown("Sample of uploaded data:")
            st.dataframe(df.head())

            if st.button("Run bulk predictions"):
                with st.spinner("Predicting..."):
                    required_len_ok = df.shape[1] >= 9
                    if not required_len_ok:
                        st.error("CSV must contain at least 9 columns in correct order. See instructions.")
                    else:
                        X = df.iloc[:, :9].values.astype(float)
                        try:
                            preds = model.predict(X)
                            probs = model.predict_proba(X)[:, 1] * 100
                        except Exception:
                            preds = model.predict(X)
                            probs = np.array(preds, dtype=float)
                        df["Predicted"] = preds
                        df["Probability (%)"] = np.round(probs, 2)
                        st.success("Bulk prediction completed")
                        st.dataframe(df.head(30))

                        user = st.session_state.username if st.session_state.logged_in else "anonymous"
                        for i in range(X.shape[0]):
                            log_prediction(user, X[i].tolist(), int(preds[i]), float(probs[i]))

                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇ Download predictions CSV", csv, "predictions.csv", "text/csv")
                        st.balloons()
        except Exception as e:
            st.error("Error processing CSV: " + str(e))

elif page == "Dashboard (Analytics)":
    st.subheader("📊 Analytics Dashboard")
    conn = get_db_conn()
    df_preds = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn, parse_dates=["timestamp"])
    conn.close()

    st.markdown("**Prediction History (most recent 50)**")
    if df_preds.empty:
        st.info("No predictions logged yet.")
    else:
        st.dataframe(df_preds.head(50))

        st.markdown("### Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        total = len(df_preds)
        att_yes = df_preds[df_preds["predicted"] == 1].shape[0]
        att_no = df_preds[df_preds["predicted"] == 0].shape[0]
        avg_prob = df_preds["probability"].mean() if not df_preds["probability"].isnull().all() else 0
        col1.metric("Total predictions", total)
        col2.metric("Predicted Attend", att_yes)
        col3.metric("Predicted Not Attend", att_no)
        col4.metric("Average probability", f"{avg_prob:.2f}%")

        st.markdown("### Charts")
        fig1, ax1 = plt.subplots(figsize=(6,3))
        sns.countplot(data=df_preds, x="predicted", ax=ax1)
        ax1.set_xticklabels(["Not Attend (0)", "Attend (1)"])
        ax1.set_title("Attendance Count")
        st.pyplot(fig1)

        st.markdown("Attendance probability distribution")
        fig2, ax2 = plt.subplots(figsize=(6,3))
        sns.histplot(df_preds["probability"].dropna(), bins=20, ax=ax2)
        st.pyplot(fig2)

elif page == "Admin Panel":
    if not st.session_state.logged_in or st.session_state.role != "admin":
        st.error("Admin only. Please login as an admin.")
    else:
        st.subheader("🛠 Admin Panel")

        st.markdown("### Upload new model / scaler")
        st.markdown("Upload a trained model (.pkl) and/or scaler (.pkl). Filenames will overwrite existing files.")
        mfile = st.file_uploader("Upload model .pkl", type=["pkl"], key="mfile")
        sfile = st.file_uploader("Upload scaler .pkl", type=["pkl"], key="sfile")
        if st.button("Upload and replace files"):
            if mfile is not None:
                with open(MODEL_PATH, "wb") as f:
                    f.write(mfile.getbuffer())
                st.success("Model file saved.")
            if sfile is not None:
                with open(SCALER_PATH, "wb") as f:
                    f.write(sfile.getbuffer())
                st.success("Scaler file saved.")
            st.cache_resource.clear()
            st.session_state["rerun_flag"] = not st.session_state.get("rerun_flag", False)
            st.experimental_set_query_params(dummy=datetime.now().timestamp())


        st.markdown("---")
        st.markdown("### Manage users")
        conn = get_db_conn()
        users_df = pd.read_sql_query("SELECT id, username, role FROM users", conn)
        st.dataframe(users_df)
        conn.close()
        new_user = st.text_input("New username (admin can create user)")
        new_pass = st.text_input("New password", type="password")
        new_role = st.selectbox("Role", ["user", "admin"])
        if st.button("Create user"):
            if new_user and new_pass:
                ok, msg = register_user(new_user.strip(), new_pass, new_role)
                if ok:
                    st.success("User created")
                    st.session_state["rerun_flag"] = not st.session_state.get("rerun_flag", False)
                    st.experimental_set_query_params(dummy=datetime.now().timestamp())

                else:
                    st.error(msg)
            else:
                st.error("Provide username and password")

        st.markdown("---")
        conn = get_db_conn()
        df_all = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn, parse_dates=["timestamp"])
        conn.close()
        st.markdown("### All prediction logs")
        st.dataframe(df_all.head(200))
        download_csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download full logs CSV", download_csv, file_name="prediction_logs.csv", mime="text/csv")

elif page == "About":
    st.subheader("ℹ About this app")
    st.markdown("""
    **Event Attendance Prediction — Pro**  
    - Login system (SQLite + password hash)  
    - Admin panel for model upload and user management  
    - Prediction logging (saved to SQLite)  
    - Single & Bulk prediction  
    - Analytics dashboard with charts  
    - Streamlit deployment ready
    """)
    st.markdown("**Developer:** Selva Krishnan")
