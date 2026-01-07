import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_loader import fetch_stock_data, fetch_news, get_local_tickers
from preprocessing import Preprocessor
from train import train_model, predict
from datetime import datetime, timedelta

# Page Config
st.set_page_config(page_title="Transformer Stock Predictor", layout="wide")

# Title
st.title("📈 Transformer-Based Stock Predictor")
st.markdown("Predict stock prices using a Transformer model with News Sentiment Analysis.")

# Check for Offline Data
local_tickers = get_local_tickers()
is_offline = len(local_tickers) > 0

# Sidebar
st.sidebar.header("Configuration")

# Ticker Selection Logic
if is_offline:
    st.sidebar.success(f"Offline Mode Active: Found {len(local_tickers)} local tickers.")
    # Add 'Custom' option to allow manual entry even if offline data exists
    ticker_options = local_tickers + ["Custom"]
    selected_ticker = st.sidebar.selectbox("Select Ticker", ticker_options)
    
    if selected_ticker == "Custom":
        ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    else:
        ticker = selected_ticker
else:
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")

timeframe = st.sidebar.selectbox("Prediction Horizon", ["1 Day", "1 Hour", "1 Minute"])

epochs = st.sidebar.slider("Training Epochs", 5, 100, 20)
seq_length = st.sidebar.slider("Sequence Length", 10, 100, 60)

# Map timeframe to yfinance arguments
if timeframe == "1 Day":
    period = "2y"
    interval = "1d"
elif timeframe == "1 Hour":
    period = "1y" # Max for 1h is 2y, keep it safe
    interval = "1h"
else: # 1 Minute
    period = "5d" # Max for 1m is 7d
    interval = "1m"

# Main Logic
if st.button("Run Prediction"):
    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_stock_data(ticker, period, interval)
        
        if df is None or df.empty:
            st.warning(f"Could not fetch live data for {ticker} (likely Rate Limit). Generating SYNTHETIC data for demonstration.")
            # Generate synthetic data
            dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
            prices = [150]
            for _ in range(499):
                change = np.random.normal(0, 2)
                prices.append(prices[-1] + change)
            df = pd.DataFrame({'Close': prices, 'Open': prices, 'High': prices, 'Low': prices, 'Volume': 1000000}, index=dates)

        if df is not None:
            # Display Data
            st.subheader(f"Historical Data ({ticker})")
            
            # Plot CandleStick
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'])])
            st.plotly_chart(fig, use_container_width=True)
            
            # Preprocessing
            with st.spinner("Processing data and analyzing sentiment..."):
                prep = Preprocessor()
                try:
                    df = prep.add_technical_indicators(df)
                except Exception as e:
                    st.error(f"Error in indicators: {e}")
                    st.stop()
                    
                news = fetch_news(ticker)
                sentiment_score = prep.get_sentiment_score(news)
                
                st.metric("News Sentiment Score", f"{sentiment_score:.2f}", delta_color="normal")
                if news:
                    with st.expander("Recent News Headlines"):
                        for item in news[:5]:
                            st.write(f"- [{item['title']}]({item['link']}) ({item['publisher']})")
            
            # Model Training
            st.subheader("Model Training & Prediction")
            
            X, y, scaler = prep.prepare_data_for_model(df, sentiment_score, seq_length=seq_length)
            
            if len(X) == 0:
                st.error("Not enough data to create sequences. Try a longer period or shorter sequence length.")
            else:
                # Train/Test Split
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                input_dim = X.shape[2]
                
                with st.spinner("Training Transformer Model..."):
                    model, loss_history = train_model(X_train, y_train, input_dim, seq_length=seq_length, epochs=epochs)
                
                st.success("Training Complete!")
                
                # Predict
                predictions = predict(model, X_test, scaler)
                
                # Inverse transform y_test for comparison
                # Similar hack for y_test as in predict function
                n_features = scaler.n_features_in_
                dummy_y = np.zeros((len(y_test), n_features))
                dummy_y[:, 0] = y_test 
                actuals = scaler.inverse_transform(dummy_y)[:, 0]
                
                # Plot Predictions
                pred_fig = go.Figure()
                # Generate time index for test set
                # The test set starts after 'split' + 'seq_length' original points
                test_indices = df.index[split + seq_length:]
                
                # Align lengths
                min_len = min(len(test_indices), len(actuals))
                test_indices = test_indices[:min_len]
                actuals = actuals[:min_len]
                predictions = predictions[:min_len]
                
                pred_fig.add_trace(go.Scatter(x=test_indices, y=actuals, mode='lines', name='Actual Price'))
                pred_fig.add_trace(go.Scatter(x=test_indices, y=predictions, mode='lines', name='Predicted Price'))
                
                st.plotly_chart(pred_fig, use_container_width=True)
                
                # Future Prediction (Next Step)
                last_sequence = X[-1].reshape(1, seq_length, input_dim)
                future_pred_scaled = model(torch.from_numpy(last_sequence).float().to(next(model.parameters()).device))
                future_pred_scaled = future_pred_scaled.cpu().detach().numpy()
                
                dummy_future = np.zeros((1, n_features))
                dummy_future[:, 0] = future_pred_scaled[:, 0]
                future_price = scaler.inverse_transform(dummy_future)[0, 0]
                
                # Adjust with sentiment (heuristic)
                # If sentiment is very positive (>0.5), boost slightly?
                # Creating a simple visual adjustment explanation
                adjusted_price = future_price * (1 + (sentiment_score * 0.001)) # Very safe adjustment
                
                st.subheader(f"Next {timeframe} Prediction")
                col1, col2 = st.columns(2)
                col1.metric("Predicted Price", f"${future_price:.2f}")
                col2.metric("Sentiment Adjusted Price", f"${adjusted_price:.2f}", f"{(adjusted_price-future_price):.2f}")
                
        else:
            st.error("Failed to fetch data.")
