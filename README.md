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
