import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 0.5rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #334155;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    .metric-box {
        background: #f8fafc;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #64748b;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s ease;
    }
    .metric-box:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-box h3 {
        margin: 0;
        color: #1e293b;
        font-weight: 600;
    }
    .metric-box p {
        margin: 0.3rem 0 0 0;
        color: #64748b;
        font-size: 0.9rem;
    }
    .stTextInput > div > div > input {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        color: #334155;
        padding: 0.5rem;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #64748b;
        box-shadow: 0 0 0 2px #64748b;
    }
    .chart-container {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #64748b;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton > button:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(100, 116, 139, 0.7); /* soft blue glow */
    background-color: #475569; /* dark blue background on focus */
    color: white;
    }
    
    .stButton > button:active {
        background-color: #334155; /* darker blue when pressed */
        box-shadow: none;
        color: white;
    }
    .stButton > button:hover {
        background-color: #475569;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_prediction_model():
    try:
        model = load_model('Stock Prediction Model.keras')
        return model
    except Exception as e:
        st.error("Model file not found or failed to load. Please ensure 'Stock Prediction Model.keras' exists.")
        return None


model = load_prediction_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>Stock Market Predictor</h1>
    <p>AI-powered stock analysis and prediction</p>
</div>
""", unsafe_allow_html=True)

# Input & Analyze button
ticker = st.text_input('Enter Stock Ticker', 'GOOG', help="e.g., AAPL, MSFT, TSLA").upper().strip()

analyze = st.button('Analyze Stock')

if not analyze:
    st.info("Enter a ticker and press 'Analyze Stock' to start.")
    st.stop()

# Define data range
start = '2012-01-01'
end = '2022-12-31'

with st.spinner(f'Loading data for {ticker}...'):
    data = yf.download(ticker, start=start, end=end, progress=False)

if data.empty:
    st.error(f"No data found for ticker '{ticker}'. Please check the symbol and try again.")
    st.stop()

# Current Price and Daily Change
current_price = float(data.Close.iloc[-1])
prev_price = float(data.Close.iloc[-2])
price_change = ((current_price - prev_price) / prev_price) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <h3>${current_price:.2f}</h3>
        <p>Current Price</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    change_color = "#059669" if price_change >= 0 else "#dc2626"
    st.markdown(f"""
    <div class="metric-box">
        <h3 style="color: {change_color};">{price_change:+.2f}%</h3>
        <p>Daily Change</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    volume = float(data.Volume.iloc[-1]) if 'Volume' in data.columns else 0
    volume_str = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.1f}K"
    st.markdown(f"""
    <div class="metric-box">
        <h3>{volume_str}</h3>
        <p>Volume</p>
    </div>
    """, unsafe_allow_html=True)

# Raw data display
st.subheader('Stock Data')
with st.expander("View Raw Data", expanded=False):
    st.dataframe(data.tail(10), use_container_width=True)

# Prepare data for prediction
data_close = data[['Close']].copy()
train_size = int(len(data_close)*0.8)

train_data = data_close[:train_size]
test_data = data_close[train_size-100:]  # Include 100 days before test for context

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_data)

train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)

# Moving averages cached
@st.cache_data
def compute_moving_averages(series):
    ma50 = series.rolling(window=50).mean()
    ma100 = series.rolling(window=100).mean()
    return ma50, ma100

ma50, ma100 = compute_moving_averages(data_close.Close)

# Plot Price vs MA50
st.subheader('Price vs 50-Day Moving Average')
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data_close.index, data_close.Close, label='Stock Price', color='#1e293b', linewidth=1.8, alpha=0.9)
ax1.plot(ma50.index, ma50, label='MA50', color='#64748b', linewidth=2.5)
ax1.set_title(f'{ticker} Stock Price vs 50-Day Moving Average', fontsize=15, fontweight='500', color='#334155')
ax1.set_xlabel('Date', color='#64748b')
ax1.set_ylabel('Price ($)', color='#64748b')
ax1.legend(frameon=False)
ax1.grid(True, alpha=0.2, color='#e2e8f0')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#e2e8f0')
ax1.spines['bottom'].set_color('#e2e8f0')
fig1.tight_layout()
st.pyplot(fig1)
st.markdown('</div>', unsafe_allow_html=True)

# Plot Price vs MA50 vs MA100
st.subheader('Price vs 50-Day & 100-Day Moving Averages')
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(data_close.index, data_close.Close, label='Stock Price', color='#1e293b', linewidth=1.8, alpha=0.9)
ax2.plot(ma50.index, ma50, label='MA50', color='#64748b', linewidth=2.5)
ax2.plot(ma100.index, ma100, label='MA100', color='#94a3b8', linewidth=2.5)
ax2.set_title(f'{ticker} Stock Price vs Moving Averages', fontsize=15, fontweight='500', color='#334155')
ax2.set_xlabel('Date', color='#64748b')
ax2.set_ylabel('Price ($)', color='#64748b')
ax2.legend(frameon=False)
ax2.grid(True, alpha=0.2, color='#e2e8f0')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#e2e8f0')
ax2.spines['bottom'].set_color('#e2e8f0')
fig2.tight_layout()
st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)

# Generate predictions
if model:
    with st.spinner('Generating predictions...'):
        x_test = []
        y_test = []
        test_scaled = np.array(test_scaled)
        for i in range(100, len(test_scaled)):
            x_test.append(test_scaled[i-100:i])
            y_test.append(test_scaled[i, 0])
        if len(x_test) == 0:
            st.warning("Not enough data for predictions.")
        else:
            x_test = np.array(x_test)
            y_test = np.array(y_test).reshape(-1,1)

            preds = model.predict(x_test, verbose=0)
            # Inverse transform
            preds_inv = scaler.inverse_transform(preds)
            y_test_inv = scaler.inverse_transform(y_test)

            # Calculate metrics
            mse = np.mean((y_test_inv - preds_inv) ** 2)
            rmse = np.sqrt(mse)
            accuracy = max(0, 100 - (rmse / np.mean(y_test_inv) * 100))

            # Next predicted price (last prediction)
            next_pred_price = preds_inv[-1][0]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>${next_pred_price:.2f}</h3>
                    <p>Next Predicted Price</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>{accuracy:.1f}%</h3>
                    <p>Model Accuracy</p>
                </div>
                """, unsafe_allow_html=True)

            # Prediction plot
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig3, ax3 = plt.subplots(figsize=(12, 6))

            # Align dates for test period (starting from train_size)
            test_dates = data_close.index[train_size:train_size + len(y_test_inv)]

            ax3.plot(test_dates, y_test_inv, label='Actual Price', color='#1e293b', linewidth=2.5, alpha=0.9)
            ax3.plot(test_dates, preds_inv, label='Predicted Price', color='#64748b', linewidth=2.5, linestyle='--', alpha=0.8)
            ax3.set_title(f'{ticker} Actual vs Predicted Prices', fontsize=15, fontweight='500', color='#334155')
            ax3.set_xlabel('Date', color='#64748b')
            ax3.set_ylabel('Price ($)', color='#64748b')
            ax3.legend(frameon=False)
            ax3.grid(True, alpha=0.2, color='#e2e8f0')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_color('#e2e8f0')
            ax3.spines['bottom'].set_color('#e2e8f0')
            plt.xticks(rotation=45)
            fig3.tight_layout()
            st.pyplot(fig3)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("Prediction model not available. Please check your model file.")

# Footer & download button
st.markdown("---")
st.markdown("""
### About This App
This application uses LSTM neural networks to analyze historical stock data and predict future price movements.

**Disclaimer**: This is for educational purposes only. Not financial advice.
""")

if st.button('Download Stock Data as CSV'):
    csv = data.to_csv()
    st.download_button(
        label="Download CSV File",
        data=csv,
        file_name=f'{ticker}_stock_data.csv',
        mime='text/csv'
    )
