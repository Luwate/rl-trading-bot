from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import time  
from pydantic import BaseModel
from stable_baselines3 import PPO
from crypto_trading_env import CryptoTradingEnv

app = FastAPI()

# Load the test dataset
file_path = "data/test_data.csv"
df = pd.read_csv(file_path, parse_dates=["Open Time"])

# Load saved preprocessing models
scaler = joblib.load("model/scaler.pkl")
pca = joblib.load("model/pca.pkl")

def preprocess_test_data(df):
    """Apply the same scaling and PCA transformation used in training"""
    _, _, df_pca = CryptoTradingEnv.preprocess_data(df, pca_components=pca.n_components)
    return df_pca

# Apply preprocessing to test data
test_df = preprocess_test_data(df)

# Load the trained RL model 
model = PPO.load("model/ppo_crypto_trading_bot_lstm.zip")

# In-memory trade storage (replace with DB later)
trade_data = []

class TradeRequest(BaseModel):
    timestamp: str
    price: float
    action: str  # 'buy' or 'sell'
    quantity: float
    portfolio_value: float
    holdings: float  
    reward: float

@app.post("/trade/")
def record_trade(trade: TradeRequest):
    """Store trade executed by the RL model."""
    trade_data.append(trade.dict())
    return {"message": "Trade recorded"}

@app.get("/trades/")
def get_trades():
    """Return all executed trades."""
    if not trade_data:
        return []  # Return an empty list instead of a message

    df = pd.DataFrame(trade_data)

    # Ensure 'timestamp' exists
    if "timestamp" not in df.columns:
        return {"error": "Timestamp column missing from trade data."}

    # Convert to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")  
    df.dropna(subset=["timestamp"], inplace=True)  

    return df.to_dict(orient="records")

@app.get("/portfolio/")
def get_portfolio():
    """Return portfolio metrics, including holdings and last timestamp."""
    if not trade_data:
        return {
            "portfolio_value": 100000, 
            "sharpe_ratio": 0, 
            "drawdown": 0, 
            "total_trades": 0, 
            "profit": 0, 
            "holdings": 0, 
            "timestamp": None
        }

    df = pd.DataFrame(trade_data)
    df['returns'] = df['portfolio_value'].pct_change()
    
    sharpe_ratio = df['returns'].mean() / df['returns'].std() * np.sqrt(252) if df['returns'].std() != 0 else 0
    drawdown = (df['portfolio_value'] / df['portfolio_value'].cummax() - 1).min()
    total_profit = df['portfolio_value'].iloc[-1] - 100000

    return {
        "portfolio_value": df['portfolio_value'].iloc[-1],
        "sharpe_ratio": round(sharpe_ratio, 2),
        "drawdown": round(drawdown, 2),
        "total_trades": len(trade_data),
        "profit": round(total_profit, 2),
        "holdings": df['holdings'].iloc[-1],  
        "timestamp": df["timestamp"].iloc[-1]  
    }

@app.post("/run_model/")
def run_model():
    """Run the RL model on the test dataset and execute trades dynamically with progressive execution."""
    global trade_data
    trade_data = []  # Reset trade history

    # Initial Portfolio Setup
    initial_balance = 100000
    balance = initial_balance
    holdings = 0
    tx_cost = 0.001 

    # Get expected shape from model
    expected_obs_shape = model.observation_space.shape  # Should be (24, 107)
    num_features = expected_obs_shape[1]  # Should be 107

    print(f"DEBUG: Model expects observation shape: {expected_obs_shape}")

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        current_price = row["BTCUSDT_Close"]

        # Ensure State Matches Model's Expected Shape
        if i >= 24:  # Start only when we have enough past data
            state = test_df.iloc[i-24:i].drop(columns=["Open Time"]).values  # Last 24 time steps
            state = state.astype(np.float32)  # Convert to float32

            # Manually add normalized balance & holdings
            norm_balance = balance / initial_balance  # Normalize balance
            norm_holdings = (holdings * current_price) / (initial_balance)  # Normalize holdings

            balance_holdings_array = np.array([[norm_balance, norm_holdings]] * 24)  # Expand for all time steps
            state = np.hstack((state, balance_holdings_array))  # Append to state

        else:
            # Handle initial steps (before 24 rows exist)
            state = np.zeros((24, num_features), dtype=np.float32)  # Fill missing data with zeros

        print(f"🔍 DEBUG: State Shape Before Prediction: {state.shape}")  # Log shape

        # Predict action
        action, _ = model.predict(state)
        trade_fraction = np.tanh(action[0])  # Convert action to (-1, 1) range

        # Default Action Setup
        action_type = "hold"
        quantity = 0  

        # Execute Buy Trade
        if trade_fraction > 0:
            cost = trade_fraction * balance * (1 + tx_cost)
            quantity = cost / (current_price * (1 + tx_cost))
            if balance >= cost:  # Ensure enough balance
                holdings += quantity
                balance -= cost
                action_type = "buy"

        # Execute Sell Trade (Only if Holdings Exist)
        elif trade_fraction < 0 and holdings > 0:
            sell_amount = min(abs(trade_fraction) * holdings, holdings)  # Avoid over-selling
            sold_value = sell_amount * current_price * (1 - tx_cost)
            balance += sold_value
            holdings -= sell_amount
            action_type = "sell"

        # Update Portfolio Value
        portfolio_value = balance + (holdings * current_price)

        # Compute Logarithmic Return (Avoid log(0) errors)
        prev_value = trade_data[-1]["portfolio_value"] if trade_data else initial_balance
        log_return = np.log(portfolio_value / prev_value) if prev_value > 0 else 0

        # Compute Reward Function
        realized_profit = balance - prev_value
        reward = (
            0.7 * log_return +  
            0.2 * (realized_profit / initial_balance) if realized_profit > 0 else 0  
            - 0.05 * (trade_fraction ** 2)  
            - 0.05 if abs(trade_fraction) < 0.1 else 0  
        )
        reward = np.clip(reward, -5, 5)  # Clip to avoid instability

        # Store Every Step (Including Holds)
        trade_data.append({
            "timestamp": row["Open Time"].strftime("%Y-%m-%d %H:%M:%S"),
            "price": current_price,
            "action": action_type,
            "quantity": quantity,
            "portfolio_value": portfolio_value,
            "holdings": holdings,  
            "reward": reward
        })

        # Delay execution for real-time effect
        time.sleep(0.5)

    print(f" Model Execution Complete: {len(trade_data)} time steps recorded.")
    print(f" Model Execution Stopped at Step: {i} | Latest Timestamp: {row['Open Time']}", flush=True)

    return {"message": f"Model executed on test data with {len(trade_data)} time steps recorded."}


