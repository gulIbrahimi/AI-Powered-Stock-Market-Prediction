# Stock Market Predictor 

Welcome to the **Stock Market Predictor**, an AI-powered web application that analyzes historical stock data and forecasts future price movements using a pre-trained LSTM neural network. This project aims to provide clear, insightful visualizations and predictions to help users better understand stock trends in an intuitive and interactive way.

---

## Features

- **Clean and modern UI** with custom styling for a smooth user experience.
- **Real-time stock data fetching** using Yahoo Finance (`yfinance`).
- **Multiple data visualizations** including stock price, moving averages (MA50, MA100), and actual vs predicted prices.
- **AI-driven price predictions** based on historical trends, powered by a pre-trained Keras LSTM model.
- **Model accuracy metrics** presented alongside predictions for transparency.
- **Downloadable stock data** for offline analysis.
- Responsive design with thoughtful color palette and smooth interaction effects.

---

## How It Works

1. You enter a valid stock ticker symbol (e.g., `AAPL`, `TSLA`).
2. The app fetches historical price data from Yahoo Finance (default range: 2012-2022).
3. It calculates moving averages and visualizes them with the raw stock prices.
4. The pre-trained LSTM model predicts future stock prices based on the recent trends.
5. Predictions and accuracy metrics are displayed with interactive charts.
6. You can download the fetched data as a CSV file for further analysis.

---

## Tech Stack

- Python 3
- [Streamlit](https://streamlit.io/) for the web interface
- [yfinance](https://pypi.org/project/yfinance/) for financial data retrieval
- [Keras](https://keras.io/) and TensorFlow for AI model
- [scikit-learn](https://scikit-learn.org/) for data scaling
- [Matplotlib](https://matplotlib.org/) for plotting charts
- Custom CSS styling integrated with Streamlit components

---

## Notes & Disclaimer
This project is intended for educational purposes only.

Stock market predictions are inherently uncertain; do not use this app as financial advice.

Model accuracy may vary depending on data quality and market conditions.

## Future Improvements
- Add user authentication and personalized watchlists.

- Integrate more advanced models with sentiment analysis.

- Extend date range flexibility and live data streaming.

- Deploy on cloud platforms with scalable backend support.
