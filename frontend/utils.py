import pandas as pd
import requests

API_URL = "http://localhost:8000"

# Fetch BTC test data
file_path = "../data/test_data.csv"
df = pd.read_csv(file_path, parse_dates=["Open Time"])
btc_df = df[["Open Time", "BTCUSDT_Close"]]

test_df = btc_df.copy()

def get_btc_test_data():
    """Returns the BTC test data for plotting."""
    return test_df

def fetch_portfolio():
    """Fetch portfolio value, Sharpe Ratio, and Drawdown from FastAPI backend."""
    try:
        response = requests.get(f"{API_URL}/portfolio/")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def fetch_trades():
    """Fetch trade data (buy/sell) from the backend."""
    try:
        response = requests.get(f"{API_URL}/trades/")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_model():
    """Trigger the RL model to execute trades."""
    try:
        response = requests.post(f"{API_URL}/run_model/")
        return response.json()
    except Exception as e:
        return {"error": str(e)}
