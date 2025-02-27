import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time  # For progressive updates
from utils import get_btc_test_data, fetch_trades, fetch_portfolio, run_model

# ---- CONFIGURE PAGE ----
st.set_page_config(page_title="BTC Trading Dashboard", layout="wide")

# ---- TITLE & DESCRIPTION ----
st.title("🤖 RL Trading Dashboard")

st.markdown("""
### 📌 About This Dashboard  
This dashboard visualizes the performance of an **AI-powered Reinforcement Learning (RL) trading bot**.  
The bot is trained to analyze historical **BTC/USDT** market data and execute trades **autonomously** to maximize portfolio returns.  

💡 **How It Works:**  
- The bot receives **historical price & indicator data** as input.  
- It makes trading decisions (**Buy / Sell / Hold**) based on **reinforcement learning algorithms**.  
- The **portfolio value, trades, and performance metrics** are updated in real-time.  

⚠️ **Disclaimer:**  
This dashboard runs on **historical test data**. The bot is not connected to a live trading account.  
Its performance in historical data does **not guarantee** profitability in real markets.  
""")

st.divider()  # Add visual separator

# ---- RUN MODEL BUTTON ----
if st.button("🚀 Run Model on Test Data"):
    with st.spinner("📈 Model is running... Please wait for updates."):
        response = run_model()
    st.success(response.get("message", "Model execution completed."))
    st.toast("📊 Model executed! Refreshing data...")  
    time.sleep(1)  # Small delay to allow model execution
    st.rerun()  # Force refresh to update trades & portfolio

# ---- FETCH DATA ----
btc_df = get_btc_test_data()
trades = fetch_trades()
portfolio = fetch_portfolio()

# ---- PROCESS TRADE DATA (BUY & SELL COUNTS) ----
num_buys, num_sells = 0, 0
if trades and "error" not in trades:
    trade_df = pd.DataFrame(trades)
    if "action" in trade_df.columns:
        num_buys = (trade_df["action"] == "buy").sum()
        num_sells = (trade_df["action"] == "sell").sum()

# ---- PORTFOLIO & HOLDINGS METRICS ----
if "error" not in portfolio:
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Portfolio Value", f"${portfolio['portfolio_value']:,.2f}")
    col2.metric("📊 Sharpe Ratio", f"{portfolio['sharpe_ratio']:.2f}")
    col3.metric("📉 Max Drawdown", f"{portfolio['drawdown']:.2%}")

    # ---- SIDEBAR: LIVE TRADING METRICS ----
    with st.sidebar:
        st.subheader("📊 Current Holdings")
        st.metric("BTC Holdings", f"{portfolio.get('holdings', 0):.6f} BTC")
        st.metric("🔄 Last Processed Time", f"{portfolio.get('timestamp', 'N/A')}")
        st.metric("🟢 Number of Buys", f"{num_buys}")
        st.metric("🔴 Number of Sells", f"{num_sells}")

# ---- CREATE CHART ----
st.subheader("📈 BTC Price vs Portfolio Performance")
fig = go.Figure()

# 📌 Plot BTC Price
fig.add_trace(go.Scatter(
    x=btc_df["Open Time"], 
    y=btc_df["BTCUSDT_Close"], 
    mode="lines", 
    name="BTC Price", 
    line=dict(color="blue", width=2)
))

# ---- ADD TRADES & FUTURE PRICES ----
if trades and "error" not in trades:
    trade_df["timestamp"] = pd.to_datetime(trade_df["timestamp"], errors="coerce")
    latest_timestamp = trade_df["timestamp"].max()

    # DEBUGGING: Check if latest_timestamp is correct
    print("DEBUG: Latest Timestamp Processed by Model:", latest_timestamp)

    if pd.isna(latest_timestamp):
        st.warning("⚠️ No valid timestamps in trade data. Future shading won't work.")

    # Ensure future data is correctly extracted
    future_data = btc_df[btc_df["Open Time"] > latest_timestamp]

    if not future_data.empty:
        fig.add_trace(go.Scatter(
            x=future_data["Open Time"], 
            y=future_data["BTCUSDT_Close"], 
            mode="lines", 
            name="Future Prices",
            line=dict(color="lightgrey", width=2, dash="dot"), 
            fill="tonexty",  
            fillcolor="rgba(211,211,211,0.3)"  
        ))

    # Portfolio Value Line
    fig.add_trace(go.Scatter(
        x=trade_df["timestamp"], 
        y=trade_df["portfolio_value"], 
        mode="lines", 
        name="Portfolio Value", 
        line=dict(color="purple", dash="dot")
    ))

    # ✅ BUY & SELL MARKERS
    buys = trade_df[trade_df["action"] == "buy"]
    sells = trade_df[trade_df["action"] == "sell"]

    fig.add_trace(go.Scatter(
        x=buys["timestamp"], 
        y=buys["price"], 
        mode="markers", 
        name="Buy",
        marker=dict(color="green", size=10, symbol="triangle-up")
    ))

    fig.add_trace(go.Scatter(
        x=sells["timestamp"], 
        y=sells["price"], 
        mode="markers", 
        name="Sell",
        marker=dict(color="red", size=10, symbol="triangle-down")
    ))

else:
    st.warning("⚠️ No trade data available. Run the model first.")

# ---- DISPLAY CHART ----
st.plotly_chart(fig, use_container_width=True)

# ---- AUTO-REFRESH LOGIC ----
#st.toast("🔄 Updating data in real-time...")
#for _ in range(30):  # Auto-refresh for 30 seconds
    #time.sleep(1)
    #st.rerun()
