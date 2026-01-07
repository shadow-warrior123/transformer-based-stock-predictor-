# Transformer-Based Stock Predictor (College Project)

🚀 **A powerful, Transformer-based stock price prediction system trained on historical data with news sentiment integration.**

## 📌 Project Overview
This project was developed for a college assignment to demonstrate the application of Transformer architectures in time-series forecasting, specifically for stock market data. It features a modern Streamlit UI, handles multiple timeframes (1 Day, 1 Hour, 1 Minute), and incorporates news sentiment analysis to refine predictions.

## ✨ Key Features
- **Transformer Model**: Uses a custom PyTorch `TimeSeriesTransformer` with Positional Encoding.
- **Sentiment Analysis**: Integrates real-time news news analysis using `FinBERT` (ProsusAI/finbert).
- **Dual Data Modes**:
  - **Live Mode**: Fetches data from Yahoo Finance API.
  - **Offline Mode**: Uses a local SQLite database (`stocks.db`) for consistent testing without API limits.
- **Interactive UI**: A sleek Streamlit dashboard for training, visualizing, and predicting.
- **GPU Ready**: Automatically detects and utilizes CUDA for accelerated training on Cloud GPUs (RunPod, Vast.ai, etc.).

## 🛠️ Technologies Used
| Component | Technology |
| :--- | :--- |
| **Model** | PyTorch (Transformer Encoder) |
| **Data** | yfinance, Pandas, NumPy |
| **Indicators** | pandas_ta (RSI, MACD, EMA) |
| **Sentiment** | HuggingFace Transformers (FinBERT) |
| **Frontend** | Streamlit |
| **Visualization** | Plotly |

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- (Optional) CUDA-enabled GPU for faster training.

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/shadow-warrior123/transformer-based-stock-predictor-
cd transformer-based-stock-predictor-

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the App
```powershell
streamlit run app.py
```

## 📈 Methodology
1. **Data Collection**: Fetches OHLCV data and recent financial news.
2. **Feature Engineering**: Calculates technical indicators like RSI and MACD.
3. **Sentiment Extraction**: Scores news headlines using a pre-trained FinBERT model.
4. **Model Training**: A Transformer model is trained on-the-fly based on user-selected parameters (Epochs, Sequence Length).
5. **Prediction**: Generates predictions for the next timestamp, adjusted by the sentiment score.

## 📊 Results & Validation
- **Convergence**: The model consistently demonstrates loss reduction (e.g., from 0.43 to 0.009) within 15-20 epochs.
- **Directional Accuracy**: Successfully predicts price direction with ~66% accuracy in backtesting on unseen data.

## 📜 License
This project is for educational purposes. Data provided by Yahoo Finance.

---
**Developed for College Project Submission.**
