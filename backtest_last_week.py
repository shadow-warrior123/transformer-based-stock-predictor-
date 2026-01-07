import torch
import numpy as np
import pandas as pd
from data_loader import fetch_stock_data
from preprocessing import Preprocessor
from train import train_model, predict
from datetime import timedelta

def backtest_last_week(ticker):
    print(f"\n{'='*20} Backtesting {ticker} on Last 7 Days {'='*20}")
    
    # 1. Fetch Data (Enough to train + test)
    df = fetch_stock_data(ticker, period='2y', interval='1d')
    
    if df is None:
        print(f"Warning: Could not fetch data for {ticker}. Generating SYNTHETIC data for backtest.")
        # Generate synthetic data with a pattern
        # Upward trend + sine wave to make it predictabl-ish
        dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='D')
        x = np.linspace(0, 50, 500)
        prices = 100 + x + 10 * np.sin(x) + np.random.normal(0, 2, 500)
        
        df = pd.DataFrame({'Close': prices, 'Open': prices, 'High': prices, 'Low': prices, 'Volume': 1000000}, index=dates)


    # 2. Add Indicators
    prep = Preprocessor()
    try:
        df = prep.add_technical_indicators(df)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return

    # 3. Split Data by Date
    # Last date in dataset
    last_date = df.index[-1]
    split_date = last_date - timedelta(days=7)
    
    print(f"Data Range: {df.index[0].date()} to {last_date.date()}")
    print(f"Training up to: {split_date.date()}")
    print(f"Testing from:   {(split_date + timedelta(days=1)).date()}")

    # Prepare sequences using ALL data first, then split the ARRAYS to ensure continuity
    # We pass 0.0 sentiment for pure price action testing
    X, y, scaler = prep.prepare_data_for_model(df, sentiment_score=0.0, seq_length=60)
    
    # We need to find the index corresponding to the split_date
    # Since X is created from rolling windows, X[i] uses data from i to i+seq_len
    # and targets y[i] which is data at i+seq_len.
    # We want y[i] to correspond to dates > split_date.
    
    # Reconstruct dates for y
    # y[0] corresponds to df index at seq_length
    y_dates = df.index[60:] 
    
    # Indices where date > split_date
    test_indices = [i for i, date in enumerate(y_dates) if date > split_date]
    
    if not test_indices:
        print("Not enough data to support this split.")
        return

    split_idx = test_indices[0]
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    test_dates = y_dates[split_idx:]
    
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples:  {len(X_test)} (Days: {[d.date() for d in test_dates]})")

    # 4. Train
    input_dim = X.shape[2]
    # Train for more epochs to ensure it learns well for this demo
    model, _ = train_model(X_train, y_train, input_dim, seq_length=60, epochs=20, batch_size=32)

    # 5. Predict
    predictions = predict(model, X_test, scaler)
    
    # Inverse transform actuals
    n_features = scaler.n_features_in_
    dummy_y = np.zeros((len(y_test), n_features))
    dummy_y[:, 0] = y_test 
    actuals = scaler.inverse_transform(dummy_y)[:, 0]

    # 6. Report
    print(f"\n{'='*5} Results for {ticker} (Last Week) {'='*5}")
    print(f"{'Date':<12} | {'Actual':<10} | {'Predicted':<10} | {'Diff %':<10}")
    print("-" * 50)
    
    correct_direction = 0
    for i in range(len(actuals)):
        date_str = test_dates[i].strftime('%Y-%m-%d')
        act = actuals[i]
        pred = predictions[i]
        diff = ((pred - act) / act) * 100
        
        # Check direction accuracy (did it move same way as actual from previous day?)
        if i > 0:
            prev_act = actuals[i-1]
            act_change = act - prev_act
            pred_change = pred - prev_act # Compare pred to PREVIOUS ACTUAL known data
            if (act_change > 0 and pred_change > 0) or (act_change < 0 and pred_change < 0):
                correct_direction += 1
        
        print(f"{date_str:<12} | {act:<10.2f} | {pred:<10.2f} | {diff:>9.2f}%")

    if len(actuals) > 1:
        dir_acc = (correct_direction / (len(actuals)-1)) * 100
        print(f"\nDirectional Accuracy: {dir_acc:.2f}%")

if __name__ == "__main__":
    # Test on a few major inputs
    backtest_last_week("SPY")
