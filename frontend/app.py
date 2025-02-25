import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time  # For progressive updates
from utils import get_btc_test_data, fetch_trades, fetch_portfolio, run_model

st.set_page_config(page_title="BTC Trading Dashboard", layout="wide")

st.title("🤖 RL Trading Dashboard")

# Run Model Button
if st.button("🚀 Run Model on Test Data"):
    with st.spinner("📈 Model is running... Please wait for updates."):
        response = run_model()
    st.success(response.get("message", "Model execution completed."))
    st.toast("📊 Model executed! Refreshing data...")  
    time.sleep(1)  # Small delay to allow model execution
    st.rerun()  # Force refresh to update trades & portfolio


# Fetch Data
btc_df = get_btc_test_data()
trades = fetch_trades()
portfolio = fetch_portfolio()

# Portfolio & Holdings Metrics
if "error" not in portfolio:
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Value", f"${portfolio['portfolio_value']:,}")
    col2.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{portfolio['drawdown']:.2%}")

    # Display Holdings and Timestamp in Sidebar
    with st.sidebar:
        st.subheader("📊 Current Holdings")
        st.metric("Holdings (BTC)", f"{portfolio.get('holdings', 0):.6f} BTC")
        st.metric("Last Processed Time", f"{portfolio.get('timestamp', 'N/A')}")

# Create Chart
fig = go.Figure()

# Plot BTC Price (Full Line)
fig.add_trace(go.Scatter(
    x=btc_df["Open Time"], 
    y=btc_df["BTCUSDT_Close"], 
    mode="lines", 
    name="BTC Price", 
    line=dict(color="blue", width=2)
))

# Fetch latest timestamp reached by the model
if trades and "error" not in trades:
    trade_df = pd.DataFrame(trades)

    if "timestamp" in trade_df.columns:
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
                name="Future Prices (Unprocessed)",
                line=dict(color="lightgrey", width=2, dash="dot"), 
                fill="tonexty",  
                fillcolor="rgba(211,211,211,0.3)"  
            ))
        else:
            print("DEBUG: No future data found. Model has processed all timestamps.")

        # Portfolio Value Line
        fig.add_trace(go.Scatter(
            x=trade_df["timestamp"], 
            y=trade_df["portfolio_value"], 
            mode="lines", 
            name="Portfolio Value", 
            line=dict(color="purple", dash="dot")
        ))

        # Buy & Sell Markers
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
        st.warning("No 'timestamp' column in trade data. Check API response.")
else:
    st.warning("No trade data available. Run the model first.")

# Display Chart
st.plotly_chart(fig, use_container_width=True)

# **Live Updates Every 1 Second (Optional)**
st.toast("🔄 Updating data in real-time...")
for _ in range(30):  # Auto-refresh for 30 seconds
    time.sleep(1)
    st.rerun()
