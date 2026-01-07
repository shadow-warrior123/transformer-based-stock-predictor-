try:
    import streamlit
    import yfinance
    import pandas_ta
    import transformers
    import plotly
    import torch
    import numpy
    print("All imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)
