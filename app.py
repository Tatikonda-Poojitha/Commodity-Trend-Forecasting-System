
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import json
import os

# ---------------- CONFIG & PERSISTENCE ----------------
st.set_page_config(page_title="Commodity Trend Forecasting", layout="wide")
DB_FILE = "users_db.json"

def load_users():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    # Default admin credentials as requested
    return {"admin": {"password": "admin123", "name": "Administrator"}}

def save_user(username, password, name):
    db = load_users()
    db[username] = {"password": password, "name": name}
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

# ---------------- SESSION INITIALIZATION ----------------
if 'auth' not in st.session_state:
    st.session_state.auth = False
    st.session_state.user_fullname = ""
    st.session_state.user_db = load_users()

# ---------------- LOGIN / REGISTER UI ----------------
if not st.session_state.auth:
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.title("📊 Commodity Trend Forecasting")
        auth_tab, reg_tab = st.tabs(["Login", "Register New User"])
        
        with auth_tab:
            with st.form("login"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if user in st.session_state.user_db and st.session_state.user_db[user]["password"] == pwd:
                        st.session_state.auth = True
                        st.session_state.user_fullname = st.session_state.user_db[user]["name"]
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with reg_tab:
            with st.form("registration"):
                new_name = st.text_input("Full Name")
                new_user = st.text_input("New Username")
                new_pwd = st.text_input("New Password", type="password")
                if st.form_submit_button("Create Account"):
                    if new_user and new_pwd:
                        save_user(new_user, new_pwd, new_name)
                        st.session_state.user_db = load_users() # Refresh local state
                        st.success("Account created! Please switch to Login tab.")
                    else:
                        st.error("Please fill all fields.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title(f"👤 {st.session_state.user_fullname}")

asset = st.sidebar.selectbox(
    "Select Asset",
    ["Gold", "Silver", "Crude Oil", "Copper", "USD-INR"]
)

ticker_map = {
    "Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F",
    "Copper": "HG=F", "USD-INR": "INR=X"
}

start_d = st.sidebar.date_input("Analysis Start Date", datetime.now() - timedelta(days=365))
end_d = st.sidebar.date_input("Analysis End Date", datetime.now())

# ---------------- DATA FUNCTION ----------------
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if data.empty: return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI'] = 100 - (100 / (1 + gain/loss))

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    
    try:
        model = ARIMA(scaled_data, order=(5, 1, 0)).fit()
        forecast_res = model.get_forecast(steps=7)
        forecast_scaled = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)

        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        lower_bound = scaler.inverse_transform(conf_int[:, 0].reshape(-1, 1)).flatten()
        upper_bound = scaler.inverse_transform(conf_int[:, 1].reshape(-1, 1)).flatten()
        
        mae = mean_absolute_error(scaled_data[-20:], model.fittedvalues[-20:])
        return data, (forecast, lower_bound, upper_bound), round(mae, 4)
    except:
        return data, None, None

result_data = load_data(ticker_map[asset], start_d, end_d)

if result_data[0] is None:
    st.error("❌ No data fetched. Check internet or date range.")
    st.stop()

df, forecast_pack, mae_acc = result_data

# ---------------- HEADER & METRICS ----------------
st.title(f"📈 {asset} Market")
symbol = "₹" if asset == "USD-INR" else "$"

c1, c2, c3, c4 = st.columns(4)
curr_price = df['Close'].iloc[-1]
c1.metric("Current Price", f"{symbol}{curr_price:.2f}")
c2.metric("Accuracy (MAE)", mae_acc)
c3.metric("52-Week High", f"{symbol}{df['Close'].max():.2f}")

if forecast_pack:
    pred_price = forecast_pack[0][-1]
    diff = pred_price - curr_price
    c4.metric("7-Day Forecast", f"{symbol}{pred_price:.2f}", delta=f"{diff:.2f}")

st.markdown("---")

# ---------------- SMART ADVISOR SECTION ----------------
st.subheader("🤖 Investment Advisor")

user_choice = st.radio(
    "What is your goal for this analysis?", 
    ["I want to Buy", "I want to Sell", "Just Monitoring"], 
    horizontal=True
)

rsi_val = df['RSI'].iloc[-1]
ma50 = df['SMA_50'].iloc[-1]

def get_verdict(choice, rsi, price, ma):
    if choice == "I want to Buy":
        if rsi < 35: return "🟢 STRONG BUY", "Asset is oversold. High probability of rebound."
        elif rsi > 65: return "🟡 WAIT", "Asset is overbought. Buying now is risky."
        else: return "🔵 ACCUMULATE", "Price is stable. Buy in small quantities."
    elif choice == "I want to Sell":
        if rsi > 65: return "🟢 SELL NOW", "Strong momentum. Perfect time to secure profits."
        elif rsi < 35: return "🔴 HOLD", "Price is at a bottom. Selling now is not advised."
        else: return "🔵 PROFIT TAKE", "Consider partial selling to de-risk."
    return "⚪ MONITORING", "Observing trend against 50-day SMA."

verdict_title, verdict_desc = get_verdict(user_choice, rsi_val, curr_price, ma50)
st.info(f"**{verdict_title}:** {verdict_desc}")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Forecast", "📏 Averages", "🔥 Volatility", "🔄 Correlation", "📋 Professional Report"
])

with tab1:
    st.subheader("7-Day Future Prediction")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'][-60:], name="Actual Price", line=dict(color="#1f77b4")))
    if forecast_pack:
        f_val, lower, upper = forecast_pack
        f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
        fig_f.add_trace(go.Scatter(x=f_dates + f_dates[::-1], y=np.concatenate([upper, lower[::-1]]), fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=True, name="95% Confidence Range"))
        fig_f.add_trace(go.Scatter(x=f_dates, y=f_val, name="Prediction", line=dict(color="orange", width=3, dash="dash")))
    fig_f.update_layout(template="plotly_dark", hovermode="x unified", height=500)
    st.plotly_chart(fig_f, use_container_width=True)

with tab2:
    st.subheader("Trend Following Indicators")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price", opacity=0.5))
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="50-Day SMA"))
    fig_ma.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name="20-Day EMA"))
    fig_ma.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig_ma, use_container_width=True)

with tab3:
    st.subheader("Market Heatmap & Momentum")
    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        df['Ret'] = df['Close'].pct_change() * 100
        df['Month_Name'] = df.index.strftime('%b')
        df['Weekday'] = df.index.strftime('%a')
        fig_heat = px.density_heatmap(df, x='Month_Name', y='Weekday', z='Ret', title="Volatility Heatmap", color_continuous_scale='RdBu')
        st.plotly_chart(fig_heat, use_container_width=True)
    with col_v2:
        st.write("**Momentum Analysis**")
        st.metric("RSI Value", f"{rsi_val:.2f}")
        if rsi_val > 70: st.warning("⚠️ Overbought")
        elif rsi_val < 30: st.success("✅ Oversold")
        else: st.info("Neutral")

with tab4:
    target = st.selectbox("Compare With", [t for t in ticker_map.keys() if t != asset])
    c_data = yf.download(ticker_map[target], start=start_d, end=end_d)
    if not c_data.empty:
        if isinstance(c_data.columns, pd.MultiIndex): c_data.columns = c_data.columns.get_level_values(0)
        combined = pd.DataFrame({asset: df['Close'], target: c_data['Close']}).dropna()
        norm = (combined - combined.min()) / (combined.max() - combined.min())
        fig_comp = px.line(norm, title="Relative Movement (0 to 1 Scale)")
        st.plotly_chart(fig_comp, use_container_width=True)

# ---------------- UPDATED: PROFESSIONAL REPORT TAB ----------------
with tab5:
    st.markdown("## 📄 Official Market Analysis Report")
    
    # 1. Profile Section
    prof_col1, prof_col2 = st.columns([1, 4])
    with prof_col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
    with prof_col2:
        st.markdown(f"**Analyst Name:** {st.session_state.user_fullname}")
        st.markdown(f"**Report ID:** {datetime.now().strftime('%Y%m%d')}-COMM-01")
        st.markdown(f"**Generation Date:** {datetime.now().strftime('%B %d, %Y | %H:%M')}")
        st.markdown(f"**Account Status:** Professional Tier (Verified)")

    st.divider()

    # 2. Executive Summary (Radio Button Info)
    st.markdown("### 🔍 Executive Summary")
    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown(f"**User Objective:** {user_choice}")
        st.markdown(f"**Target Asset:** {asset} ({ticker_map[asset]})")
    with summary_col2:
        st.markdown(f"**Recommendation:** {verdict_title}")
        st.write(f"*Analysis Detail: {verdict_desc}*")

    # 3. Technical Data Table
    st.markdown("### 📊 Market Metrics Summary")
    report_df = pd.DataFrame({
        "Metric": ["Current Price", "RSI (Momentum)", "7-Day AI Forecast", "50-Day Moving Average", "Accuracy (MAE)"],
        "Value": [f"{symbol}{curr_price:.2f}", f"{rsi_val:.2f}", f"{symbol}{forecast_pack[0][-1]:.2f}", f"{symbol}{ma50:.2f}", f"{mae_acc}"]
    })
    st.table(report_df)

    # 4. Conclusion
    st.markdown("### 📝 Conclusion & Action Plan")
    st.success(f"Based on the analysis for your goal to **{user_choice.lower()}**, the system has determined a **{verdict_title}** status for {asset}. "
             f"The technical indicators show a trend against the 50-day SMA, with a 7-day target of {symbol}{forecast_pack[0][-1]:.2f}. "
             f"Recommended Strategy: {verdict_desc}")

    # 5. Export Button
    report_text = f"""
    GLOBAL ASSET OFFICIAL REPORT
    --------------------------------------
    ANALYST: {st.session_state.user_fullname}
    DATE: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    ASSET: {asset}
    
    USER GOAL: {user_choice}
    SYSTEM VERDICT: {verdict_title}
    STRATEGY: {verdict_desc}
    
    TECHNICAL STATS:
    - Current Price: {curr_price:.2f}
    - RSI Momentum: {rsi_val:.2f}
    - 7-Day Target: {forecast_pack[0][-1]:.2f}
    - Model Accuracy (MAE): {mae_acc}
    
    --------------------------------------
    End of Professional Report
    """
    
    st.download_button(
        label="📥 Download Professional Report",
        data=report_text,
        file_name=f"Report_{asset}_{st.session_state.user_fullname}.txt",
        mime="text/plain"
    )

# ---------------- LOGOUT ----------------
if st.sidebar.button("Logout"):
    st.session_state.auth = False
    st.rerun()


