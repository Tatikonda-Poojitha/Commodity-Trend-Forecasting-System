# Commodity Trend Forecasting System

## Overview

The Commodity Trend Forecasting System is a Python-based web application that analyzes historical commodity prices and predicts future market trends using the ARIMA time series forecasting model. The application provides interactive visualizations, technical indicators, and investment recommendations through a simple Streamlit interface.

---

## Features

- User Login and Registration
- Real-time commodity price analysis using Yahoo Finance
- 7-day price forecasting using ARIMA
- Technical indicators:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
- Interactive charts using Plotly
- Smart investment recommendations (Buy / Sell / Monitor)
- Commodity comparison dashboard
- Downloadable market analysis report

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- yFinance
- Statsmodels (ARIMA)
- Scikit-learn

---

## Supported Commodities

- Gold
- Silver
- Crude Oil
- Copper
- USD/INR Exchange Rate

---

## Project Structure

```
Commodity-Trend-Forecasting-System/
│── app.py
│── users_db.json
│── requirements.txt
│── README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Tatikonda-Poojitha/Commodity-Trend-Forecasting-System.git
```

### Navigate to the project directory

```bash
cd Commodity-Trend-Forecasting-System
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## How It Works

1. Login or create a new account.
2. Select a commodity and analysis period.
3. Fetch historical market data from Yahoo Finance.
4. Calculate technical indicators (SMA, EMA, RSI).
5. Forecast future prices using the ARIMA model.
6. Visualize trends through interactive charts.
7. View investment recommendations.
8. Download the generated market analysis report.


---

## Future Enhancements

- Database integration using MySQL
- Portfolio management
- Email notifications
- PDF report generation
- Cloud deployment (AWS/Streamlit Cloud)
- Support for additional forecasting models

---

## Author

**Tatikonda Poojitha**

- **Email:** tatikondapoojitha54@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/poojitha-tatikonda-6086b62a6/
- **GitHub:** https://github.com/Tatikonda-Poojitha

---

## License

This project is intended for educational and academic purposes.