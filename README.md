# Stock Market Predictor 📈

An interactive machine learning application for forecasting stock prices using deep learning. Built with Streamlit, this project allows users to analyze historical price data, train an LSTM model directly, and visualize predictions in a clean, intuitive interface. The goal? Bring ML to finance in a way that’s both accessible and grounded in real data.

---

## 🔍 What It Does

- **Fetches real historical stock data** using `yfinance`
- **Trains a deep learning model (LSTM)** from scratch using historical closing prices
- **Visualizes predictions vs. actual performance** over time
- **Shows key financial metrics** (moving averages, price trends, error rates)
- **Exports stock data** for further analysis

This isn't just a tool to view prices — it's a full ML pipeline that starts with raw data and ends with a trained model and actionable visual feedback.

---

## ✨ Why I Built It

I wanted to go beyond black-box dashboards and instead build something grounded in machine learning fundamentals. This project reflects a growing mastery of data science, deep learning, and clean user-focused design — and a commitment to making technical tools usable, purposeful, and beautiful.

---

## 🧠 Model Overview

- **Model Type:** LSTM (Long Short-Term Memory)
- **Framework:** Keras (TensorFlow backend)
- **Inputs:** Last 100 days of scaled closing prices
- **Target:** Predict next-day closing price
- **Loss Function:** Mean Squared Error
- **Training:** Model is trained live on each stock's time series

The model captures temporal patterns in the price history to learn how trends evolve — especially suited for sequential data like finance.

---

## 🛠️ Tech Stack

- `Python 3.10+`
- `Streamlit` for UI
- `yfinance` for real-time stock data
- `NumPy`, `Pandas` for data wrangling
- `scikit-learn` for preprocessing
- `TensorFlow/Keras` for deep learning
- `Matplotlib` for plotting

---

## 📊 Visualizations

- Raw closing prices
- 50-day and 100-day moving averages
- Predicted vs. actual prices
- Daily change metrics
- Model performance (RMSE and a derived “accuracy” metric)

All visualized in a well-styled, accessible layout with downloadable options for offline analysis.

---

## Requirements
streamlit

yfinance

numpy

pandas

matplotlib

scikit-learn

tensorflow

keras

## Disclaimer
This app is for educational and experimental purposes only. It is not a financial advisory tool. Always do your own research before making investment decisions.

## About Me
Flora Ibrahimi

Rising Computer Science senior | AI Instructor | Full-Stack Developer

Focused on mastering machine learning, building purposeful tools, and building tech for good!

## Let’s Connect! 
If you’re curious about the project, want to collaborate, or just want to geek out over model architectures, I’d love to hear from you.
