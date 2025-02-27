## RL-BASED CRYPTO-CURRENCY TRADING BOT

## INTRODUCTION

Algorithmic trading has transformed financial markets, but traditional strategies struggle to adapt to changing conditions. 
Reinforcement Learning (RL) offers a solution by enabling trading bots to learn from market data and optimize strategies in real-time.
This project explores an RL-based trading bot that dynamically adjusts decisions to maximize profits while managing risk. We cover key components like environment modeling, reward functions, and backtesting. By leveraging AI, our bot aims to outperform conventional strategies, providing a smarter, more adaptive approach to trading.

## BUSINESS UNDERSTANDING

Problem Statement:
Traditional algorithmic trading relies on fixed rules, struggling to adapt to volatile markets. This leads to missed opportunities, poor risk management, and reduced profitability.

Key Challenges
1. Market Volatility – Static strategies fail in unpredictable conditions.
2. Suboptimal Decisions – Fixed models can’t learn or adapt.
3. Inefficient Risk Management – Poor balance between risk and reward.
4. High Costs – Manual strategy adjustments require oversight.
5. Competitive Disadvantage – AI-driven firms outperform traditional traders.

## OBJECTIVE

Our goal is to build an algorithmic trading bot that is trained in reinforcement learning and learns from market data to optimize its trading strategies. The bot will work to improve profits, reduce risks, and constantly adjust to the ever-changing market environment.

Our RL trading bot would appeal to:
1. Investors & Traders - The bot utilizes adaptive trading strategies that may outperform the traditional method
2. Executives & Decision-Makers - AI-powered trading solutions improve efficiency, reduce human bias, and increase market opportunity.
3. Quantitative Traders - Focus on algorithmic optimization and performance metrics.
4. Risk Management Teams - Ensure the strategy aligns with financial risk controls.

## DATA UNDERSTANDING
DATA COLLECTION

Market cryptocurrency data obtained from Binance and CMC served our purpose. This goes back to February 2020. The top 50 coins by market cap, which also describe best the volatility in the market, were chosen.
The dataset contains 10711 rows and 75 columns.
The columns contain the closing price and volume traded for each coin.
Each entry is a record of the close and volume after 4 hours.
Closing price - helps identify stable trends, reduces noise, and simplifies trade execution
Volume - confirms price movements, detects breakouts, and provides liquidity that is favorable for trades to be executed.

## VOLATILITY COMPARISON

* DOGE has high volatility which indicates instability in the market.
* USD however has the lowest volatility indicating stability in the market.
* High volatility means greater potential for both high returns and high risk. Lower volatility suggests more stability but potentially lower returns

## CORRELATION MATRIX OF CRYPTOCURRENCY RETURNS
This heatmap represents the correlation matrix of the daily returns of different cryptocurrencies. Here are the key takeaways:
* Most cryptocurrencies are positively correlated, meaning they move together in price.
* Stablecoins act as a hedge, with a very low correlation to other cryptos.
* Diversification within crypto is limited, as many assets tend to follow the same trends.

## DATA PREPARATION
FEATURE ENGINEERING
Technical indicators were feature-engineered for the RL model to optimize learning efficiency, refine decision-making, and minimize noise.
For each coin, the indicators generated include:
* Exponential Moving Average - smoothes closing price data and gives more weight to recent prices, making the model more responsive to price change.
* Moving Average Convergence Divergence - measures two moving averages and identifies trends, momentum shifts, and potential buy/sell signals.
* Relative Strength Index - measures the speed and change of price movement. This indicator helps identify overbought and oversold conditions in the market.
* Bollinger Bands - aids in identifying price volatility and potential overbought and oversold conditions.
* On-Balance Volume - helps identify trends and confirm price movements.
* Money Flow Index - Similar to RSI but considers volume, helping detect buying/selling pressure.

## PCA
Standardization was first performed on the data after which, Principle Component Analysis was employed to reduce the dimensionality of the dataset’s financial indicators. We chose to retain 89% of the variance.
This was done to avoid overfitting and optimize efficiency in the RL bot

## RL TRADING BOT
TRADING ENVIRONMENT

Gym was employed, to create a custom environment to train and test the bot.
The environment: a simulation where a trading bot learns to make profitable trading decisions.
The environment accepts a user's investment amount as input and returns the final portfolio value once the bot reaches a predefined threshold.

## REWARD FUNCTION
The primary mechanism for evaluating the quality of actions, and determining what constitutes a favorable or unfavorable decision within the trading environment.
Our reward function
* Encourages buying to stop the bot from simply holding
* Rewards for realized profits
* Punish big trades - big trades are risky so they should be disincentivized
* Inactivity penalty: It’s a trading bot, not an investor. It should avoid sitting on its hands

## RPPO
We chose a PPO model because it handles stochastic and noisy markets well, allows us to implement a dynamic range of actions for our bot, and learns relatively faster than other models in backtesting.
RPPO is ideal for algorithmic trading, where price movements and market indicators depend on past trends. RPPO was considered as it implements additional regularization techniques to improve stability and performance.
* Smoother policy updates prevent drastic policy changes.
* Liquidation is prevented while exploration and exploitation are encouraged. Gamma set to
The policy used LSTM, ‘remembers’ the previous week's data when considering the next action to take

## CONCLUSION
The RL trading bot operates effectively within predefined parameters, ensuring successful backtesting through historical data while preventing liquidation.
This is highly significant to risk management teams. This shows that the bot mitigates risk in unseen territory and adapts to different market conditions. Avoiding bad trades and managing drawdowns.
The bot is encouraged to explore multiple strategies that would be useful to Hedge Fund Executives. AI-based trading agents develop dynamic strategies that perform well in the highly volatile cryptocurrency market.
Our trading bot serves as a baseline from which a more robust algorithm could evolve into a profitable system or be integrated into a broader quant trading framework with further refinements.

## THANK YOU!
