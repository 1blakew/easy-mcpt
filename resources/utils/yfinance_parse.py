import yfinance as yf
import pandas as pd
import os

data = pd.DataFrame()
btc_usd = yf.Ticker('BTC-USD')
temp = yf.Ticker('BTC-USD').history('5y')

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
temp.to_csv('data/test_data_usd_btc.csv')