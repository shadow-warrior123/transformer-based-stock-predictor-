import pandas as pd
import sqlite3
import os

def ingest_data(excel_path, db_path='stocks.db'):
    print(f"Reading {excel_path}...")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # Clean column names
    df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '')
    
    # Standardize names
    rename_map = {
        'TICKER': 'Ticker',
        'PER': 'Period',
        'DATE': 'Date',
        'TIME': 'Time',
        'OPEN': 'Open',
        'HIGH': 'High',
        'LOW': 'Low',
        'CLOSE': 'Close',
        'VOL': 'Volume',
        'OPENINT': 'OpenInterest'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Create Datetime index
    # Format appears to be YYYYMMDD and Time is likely HHMM or minutes?
    # Sample Time: 0, 500, 1000, 1500. This looks like HHMM format (00:00, 05:00... wait 500 is 05:00?)
    # Let's inspect the Time column more closely in the app, but for now we'll store as is.
    # Actually, let's try to convert to a proper Datetime string for the DB.
    
    def parse_datetime(row):
        date_str = str(row['Date'])
        time_str = str(int(row['Time'])).zfill(4) # 500 -> 0500
        return pd.to_datetime(f"{date_str} {time_str}", format="%Y%m%d %H%M")

    print("Processing timestamps...")
    try:
        df['Datetime'] = df.apply(parse_datetime, axis=1)
    except Exception as e:
        print(f"Warning: Could not parse timestamps perfectly ({e}). Using row index as proxy if needed.")
        df['Datetime'] = pd.to_datetime(df['Date'], format='%Y%m%d') # Fallback

    # Connect to DB
    conn = sqlite3.connect(db_path)
    
    # Write to table
    print("Writing to database...")
    df.to_sql('stock_data', conn, if_exists='replace', index=False)
    
    # Create Index
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_data (Ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_data (Datetime)")
    
    conn.commit()
    conn.close()
    
    print(f"Successfully created {db_path} with {len(df)} records.")
    print(f"Available Tickers: {df['Ticker'].unique()}")

if __name__ == "__main__":
    ingest_data('Book1.xlsx')
