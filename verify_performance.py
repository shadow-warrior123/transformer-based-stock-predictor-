import torch
import numpy as np
import pandas as pd
from data_loader import fetch_stock_data
from preprocessing import Preprocessor
from train import train_model, predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

def evaluate_stock(ticker, epochs=10):
    print(f"\n{'='*20} Evaluating {ticker} {'='*20}")
    
    # 1. Fetch Data
    # Using 2y daily data as per standard '1 Day' setting
    df = fetch_stock_data(ticker, period='2y', interval='1d')
    
    if df is None or len(df) < 100:
        print(f"Warning: Could not fetch data for {ticker}. Generating SYNTHETIC data for demonstration.")
        # Generate synthetic data
        dates = pd.date_range(start="2023-01-01", periods=500, freq='D')
        # Random walk with trend
        prices = [150]
        for _ in range(499):
            change = np.random.normal(0, 2)
            prices.append(prices[-1] + change)
            
        df = pd.DataFrame({'Close': prices, 'Open': prices, 'High': prices, 'Low': prices, 'Volume': 1000000}, index=dates)


    # 2. Preprocess
    prep = Preprocessor()
    try:
        df = prep.add_technical_indicators(df)
        # Note: We skip sentiment for this pure numeric benchmark to isolate model performance
        # passing 0.0 as neutral sentiment
        X, y, scaler = prep.prepare_data_for_model(df, sentiment_score=0.0, seq_length=60)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return

    # 3. Train/Test Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training Data Size: {len(X_train)} sequences")
    print(f"Test Data Size: {len(X_test)} sequences")

    # 4. Train
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")
    
    input_dim = X.shape[2]
    # Reduced epochs for quick verification, but enough to see convergence
    model, history = train_model(X_train, y_train, input_dim, seq_length=60, epochs=epochs, batch_size=32)
    
    duration = time.time() - start_time
    print(f"Training took: {duration:.2f} seconds")

    # 5. Predict & Evaluate
    predictions = predict(model, X_test, scaler)
    
    # Inverse transform actuals
    n_features = scaler.n_features_in_
    dummy_y = np.zeros((len(y_test), n_features))
    dummy_y[:, 0] = y_test 
    actuals = scaler.inverse_transform(dummy_y)[:, 0]

    # Metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"\n--- Results for {ticker} ---")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f} (Average error per share price)")
    print(f"R2 Score: {r2:.4f} (1.0 is perfect)")
    
    return r2

if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'NVDA']
    results = {}
    
    print("Starting Bulk Verification...")
    for t in tickers:
        r2 = evaluate_stock(t, epochs=15) # 15 Epochs to balance speed/results
        results[t] = r2
        
    print("\n\nGlobal Summary (R2 Scores):")
    for t, score in results.items():
        if score is not None:
            print(f"{t}: {score:.4f}")
