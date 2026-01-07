packages = ['streamlit', 'yfinance', 'pandas_ta', 'transformers', 'plotly', 'torch', 'numpy']
for p in packages:
    try:
        __import__(p)
        print(f"{p}: OK")
    except ImportError:
        print(f"{p}: MISSING")
