import gym
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from gym import spaces
import joblib  # For saving and loading scalers

class CryptoTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000, window_size=24, log_interval=50, max_steps=None, pca_components=89):
        super(CryptoTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.log_interval = log_interval
        self.rewards = []
        self.balances = []
        self.returns = []
        self.pca_components = pca_components

        # Apply the SAME preprocessing for training data
        self.scaler, self.pca, self.df_pca = self.preprocess_data(self.df, pca_components)

        # Save the scaler and PCA for later use
        joblib.dump(self.scaler, "model/scaler.pkl")
        joblib.dump(self.pca, "model/pca.pkl")

        # Identify columns
        self.price_columns = [col for col in df.columns if col.endswith("_Close")]

        # Normalize Balance & Holdings
        self.balance_scale = initial_balance
        self.holdings_scale = df[self.price_columns].max().max()

        # Define Action Space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Define Observation Space
        num_features = self.df_pca.shape[1] - 1  # Exclude "Open Time"
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, num_features + 2), dtype=np.float32
        )

        self.max_steps = max_steps if max_steps is not None else len(self.df_pca) - self.window_size
        self.reset()

    @staticmethod
    def preprocess_data(df, pca_components=89):
        """Applies scaling and PCA transformation to input data"""
        price_columns = [col for col in df.columns if col.endswith("_Close")]
        indicator_columns = [col for col in df.columns if col not in price_columns + ["Open Time"]]

        # Standardize Data
        scaler = StandardScaler()
        all_features = price_columns + indicator_columns
        scaled_data = scaler.fit_transform(df[all_features])

        # Apply PCA to retain variance
        pca = PCA(n_components=pca_components)
        reduced_data = pca.fit_transform(scaled_data)

        # Create a new DataFrame with PCA components
        pca_columns = [f"PCA_{i+1}" for i in range(pca.n_components_)]
        df_pca = pd.DataFrame(reduced_data, columns=pca_columns)
        df_pca["Open Time"] = df["Open Time"]
        df_pca[price_columns] = df[price_columns]  # Keep original price data

        return scaler, pca, df_pca

    def _next_observation(self):
        """Fetch the last `window_size` steps as observation."""
        obs = self.df_pca.iloc[self.current_step - self.window_size:self.current_step].drop("Open Time", axis=1).values
        obs = np.array(obs, dtype=np.float32)

        # Normalize balance & holdings
        norm_balance = self.balance / self.balance_scale
        norm_holdings = (self.holdings * self.df_pca.iloc[self.current_step]["BTCUSDT_Close"]) / self.holdings_scale

        obs = np.hstack((obs, np.array([[norm_balance, norm_holdings]] * self.window_size)))
        return obs

    def reset(self):
        """Resets the environment."""
        self.current_step = self.window_size  
        self.balance = self.initial_balance
        self.holdings = 0
        self.rewards = []
        self.balances = []
        self.returns = []
        self.prev_value = self.initial_balance  # Track previous portfolio value
        self.port_value = self.initial_balance  # ✅ Initialize portfolio value
        return self._next_observation()

    def step(self, action):
        """Take action, update portfolio, and return reward."""
        current_price = self.df_pca.iloc[self.current_step]["BTCUSDT_Close"]
        done = False  

        # Tanh-scaling to avoid extreme trades
        trade_fraction = np.tanh(action[0])  

        # Transaction cost (0.1% per trade)
        tx_cost = 0.001  

        # Track previous portfolio value
        prev_portfolio_value = self.port_value  

        # Execute Trade
        if trade_fraction > 0:  # Buy
            cost = trade_fraction * self.balance * (1 + tx_cost)
            self.holdings += cost / (current_price * (1 + tx_cost))
            self.balance -= cost
        elif trade_fraction < 0 and self.holdings > 0:  # Sell only if it has holdings
            sell_amount = min(abs(trade_fraction) * self.holdings, self.holdings)
            if sell_amount > 0:  # Ensure it's worth selling
                sold_value = sell_amount * current_price
                self.balance += sold_value * (1 - tx_cost)
                self.holdings -= sell_amount

        # Compute New Portfolio Value
        portfolio_value = self.balance + (self.holdings * current_price)
        self.port_value = portfolio_value

        # Compute Logarithmic Return
        daily_return = np.log(portfolio_value / prev_portfolio_value)

        # **New Reward Function**
        realized_profit = self.balance - prev_portfolio_value  # Profit from selling
        unrealized_profit = self.port_value - prev_portfolio_value  # Change in total portfolio value
        position_size = abs(trade_fraction)

        # Price difference from the last 24-hour window
        past_price = self.df_pca.iloc[self.current_step - 1]["BTCUSDT_Close"]
        price_change = (current_price - past_price) / past_price

        # Encourage buying if price is low relative to past
        buy_bonus = 0.1 * (-price_change) if trade_fraction > 0 else 0

        reward = 1.0 * daily_return  # Increase weighting on return
        reward += 0.5 * (realized_profit / self.initial_balance) if realized_profit > 1 else 0
        reward -= 0.03 * (action[0] ** 2)  # Smaller penalty for action size
        reward -= 0.02 if abs(action[0]) < 0.1 else 0  # Reduce inaction penalty
        reward += 0.2 * buy_bonus  # Boost incentive for buying at dips

        reward = np.clip(reward, -5, 5)  # Prevent extreme values

        self.rewards.append(reward)
        self.balances.append(self.balance)

        # Terminate if bankruptcy occurs
        risk_of_ruin = (self.balance + (self.holdings * current_price)) < 0.05 * self.initial_balance and self.balance < 0.1 * self.initial_balance
        if self.current_step >= self.max_steps or risk_of_ruin:
            done = True
            print(f"❌ Episode Terminated: Bankruptcy at Step {self.current_step} | Portfolio Value: {self.port_value:.2f}")

        # Log Every Few Steps
        if self.current_step % self.log_interval == 0 or done:
            print(
                f"📊 Step {self.current_step}, Action: {action[0]:.4f}, Reward: {reward:.4f}, "
                f"Balance: {self.balance:.2f}, Holdings: {self.holdings:.6f}, Portfolio Value: {self.port_value:.2f}"
            )

        self.current_step += 1
        return self._next_observation(), reward, done, {}

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe Ratio using rolling returns."""
        if len(returns) < 5:
            return -1  
        excess_returns = np.array(returns[-10:]) - risk_free_rate
        std = max(excess_returns.std(), 1e-4)
        sharpe = excess_returns.mean() / std
        return np.clip(sharpe, -3, 3)

    def get_rewards(self):
        return self.rewards

    def get_balances(self):
        return self.balances
