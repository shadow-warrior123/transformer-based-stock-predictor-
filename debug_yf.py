import yfinance as yf
try:
    print("Attempting to download AAPL...")
    dat = yf.download("AAPL", period="1mo")
    print(dat.head())
except Exception as e:
    print(f"Error: {e}")
