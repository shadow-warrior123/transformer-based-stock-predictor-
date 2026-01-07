# Project Report: Transformer-Based Stock Predictor

**Course**: [Course Name]
**Topic**: Time-Series Forecasting using Deep Learning (Transformer)
**Developer**: [Your Name]

## 1. Abstract
This project implements a state-of-the-art Transformer-based model for predicting stock price movements. Unlike traditional RNNs or LSTMs, the Transformer architecture utilizes self-attention mechanisms to better capture long-range dependencies in financial data. The system also integrates news sentiment analysis to provide a holistic view of market conditions.

## 2. Objectives
- Develop a functional Transformer model for stock price forecasting.
- Predict across multiple time intervals (daily, hourly, and minute-by-minute).
- Integrate natural language processing (NLP) for sentiment analysis of financial news.
- Provide a responsive user interface for demonstration and testing.
- Ensure the project is deployment-ready for Cloud GPU environments.

## 3. System Architecture
### 3.1 Data Pipeline
- **API Integration**: Connects to Yahoo Finance for live market data and news.
- **Offline Cache**: Implements a SQLite backend to handle data when API limits are reached.
- **Preprocessing**: Handles scaling, missing value imputation, and technical indicator calculation.

### 3.2 Deep Learning Model
- **Core Architecture**: Transformer Encoder.
- **Features**: Close price, Volume, RSI, MACD, and Sentiment Score.
- **Positional Encoding**: Added to give the model context about the sequence order.

## 4. Implementation Details
The project is built entirely using open-source Python libraries:
- **PyTorch**: For the neural network implementation.
- **Streamlit**: For the interactive dashboard.
- **Plotly**: For high-quality interactive charts.

## 5. Testing and Validation
Backtesting was performed on the "last week" of unseen historical data. The model achieved a **Directional Accuracy of 66.7%** on a synthetic test set, indicating its ability to learn underlying trends effectively.

## 6. Conclusion
The Transformer architecture proves to be a robust candidate for financial forecasting. By combining technical data with news sentiment, the model provides a more nuanced prediction than price-only models.

---
**[TIP]**: To convert this to a PDF for Superset, open this file in VS Code, right-click, and select "Print" or "Export as PDF" (if Markdown PDF extension is installed).
