import yfinance as yf
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta

DB_PATH = 'stocks.db'

def get_local_tickers():
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        tickers = pd.read_sql("SELECT DISTINCT Ticker FROM stock_data", conn)
        conn.close()
        return tickers['Ticker'].tolist()
    except:
        return []

def fetch_stock_data(ticker, period='2y', interval='1d'):
    """
    Fetches stock data. Prioritizes local SQLite DB. Falls back to yfinance (with potential rate limits).
    """
    # 1. Check Local DB
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        # Check if ticker exists in DB
        # Note: Tickers in DB might include .V extension etc.
        # We try exact match first
        query = f"SELECT * FROM stock_data WHERE Ticker = '{ticker}' ORDER BY Datetime"
        df = pd.read_sql(query, conn)
        conn.close()

        if not df.empty:
            print(f"Loaded {len(df)} records for {ticker} from local database.")
            # Set index to Datetime
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
            return df
        else:
            print(f"Ticker {ticker} not found in local DB. Trying yfinance...")

    # 2. Fallback to yfinance (Live)
    print(f"Fetching live data for {ticker}...")
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, multi_level_index=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def fetch_news(ticker):
    """
    Fetches news. 
    NOTE: Book1.xlsx does NOT have news. This will still rely on yfinance, 
    but for the offline requirement we might just return empty content or mock news if needed.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        formatted_news = []
        if news:
            for item in news:
                formatted_news.append({
                    'title': item.get('title'),
                    'publisher': item.get('publisher'),
                    'link': item.get('link'),
                    'providerPublishTime': item.get('providerPublishTime')
                })
        return formatted_news
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

if __name__ == "__main__":
    # Test
    df = fetch_stock_data("AAPL", period="1mo", interval="1d")
    print(df.head())
    news = fetch_news("AAPL")
    print(f"Fetched {len(news)} news items.")
