#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install --upgrade yfinance')


# # About Dataset
# 
# NASDAQ Stock Data with Economic Indicators
# Overview
# This dataset comprises historical stock price data for NASDAQ-listed companies, combined with a selection of key economic indicators. It is designed to provide a comprehensive view of market behavior, facilitating financial analysis and predictive modeling. Users can explore relationships between stock performance and various economic factors.
# 
# Features
# The dataset includes the following features:
# 
# Date: The date of the recorded stock prices (formatted as YYYY-MM-DD).
# 
# Open: The price at which the stock opened for trading on a given day.
# 
# High: The highest price reached by the stock during the trading day.
# 
# Low: The lowest price recorded during the trading day.
# 
# Close: The price at which the stock closed at the end of the trading day.
# 
# Volume: The total number of shares traded during the day.
# 
# Interest Rate: The prevailing interest rate, which influences economic activity and stock performance.
# 
# Exchange Rate: The exchange rate for the USD against other currencies, reflecting international market influences.
# 
# VIX: The Volatility Index, a measure of market risk and investor sentiment, often referred to as the "fear index."
# 
# Gold: The price of gold per ounce, which serves as a traditional safe-haven asset and is often inversely correlated with stock prices.
# 
# Oil: The price of crude oil, an essential commodity that influences various sectors, especially transportation and manufacturing.
# 
# TED Spread: The difference between the interest rates on interbank loans and short-term U.S. government debt, which indicates credit risk in the banking system.
# 
# EFFR (Effective Federal Funds Rate): The interest rate at which depository institutions lend reserve balances to other depository institutions overnight, influencing overall economic activity.
# 
# Use Cases
# This dataset is suitable for a variety of applications, including:
# 
# Financial Analysis: Evaluate historical trends in stock prices relative to economic indicators.
# Predictive Modeling: Develop machine learning models to forecast stock price movements based on historical data and economic variables.
# Time Series Analysis: Conduct analyses over different time frames (daily, weekly, monthly, yearly) to identify patterns and anomalies.
# Data Source
# The data is sourced from reputable financial APIs and databases:
# 
# Yahoo Finance: Historical stock prices.
# Federal Reserve Economic Data (FRED): Economic indicators such as interest rates and VIX.
# Alpha Vantage / Quandl: Commodity prices for gold and oil.
# Conclusion
# This dataset provides a rich foundation for analysts, researchers, and data scientists interested in the intersection of stock market performance and macroeconomic conditions. Its structured features and comprehensive nature make it a valuable resource for both academic and practical financial inquiries.

# In[8]:


df=pd.read_csv("C:/Users/Pavithra/Downloads/archive (22)/nasdq/nasdq.csv")
df.head(3)


# In[11]:


df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "InterestRate", "ExchangeRate", "VIX", "TEDSpread", "EFFR", "Gold", "Oil"]

df.head(5)


# In[12]:


print(df.isnull().sum())
df.describe()


# In[16]:


# Fill or drop missing values if needed
df = df.dropna()

# Plot closing price
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='NASDAQ Closing Price')
plt.title("NASDAQ Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()


# In[17]:


df['Month'] = df.index.month
monthly_avg = df.groupby('Month')['Close'].mean()

plt.figure(figsize=(10, 5))
monthly_avg.plot(marker='o', color='purple')
plt.title("🗓️ Monthly Average Closing Price")
plt.xlabel("Month")
plt.ylabel("Average Close")
plt.grid(True)
plt.xticks(range(1, 13))
plt.show()


# In[18]:


get_ipython().system('pip install pmdarima')


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Split last 30 days as test set
train = df['Close'][:-30]
test = df['Close'][-30:]

# Auto ARIMA to find best parameters
model = auto_arima(train, seasonal=False, trace=True, suppress_warnings=True)
model.summary()

# Forecast 30 steps
forecast = model.predict(n_periods=30)
forecast_index = test.index

# Convert to pandas Series
forecast_series = pd.Series(forecast, index=forecast_index)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Actual Data')
plt.plot(forecast_series, label='Forecasted Data')
plt.title("ARIMA Forecast vs Actual")
plt.legend()
plt.grid(True)
plt.show()

rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
print(f"ARIMA Model Performance:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}")


# In[25]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Use 'Close' prices
train = df['ExchangeRate'][:-30]
test = df['ExchangeRate'][-30:]

# Fit SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

# Forecast
sarima_forecast = sarima_model.forecast(steps=30)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Actual')
plt.plot(sarima_forecast, label='SARIMA Forecast')
plt.legend()
plt.title("SARIMA Forecast vs Actual")
plt.grid(True)
plt.show()


# In[21]:


get_ipython().system('pip install prophet')


# In[32]:


from prophet import Prophet

# Prepare data for Prophet
df_prophet = df.reset_index()[['Date', 'Gold']]
df_prophet.columns = ['ds', 'y']

# Initialize and fit model
prophet_model = Prophet()
prophet_model.fit(df_prophet)

# Make future dataframe (30 days)
future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)

# Plot forecast
fig1 = prophet_model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()


# In[23]:


get_ipython().system('pip install tensorflow')


# In[24]:


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

# Create sequences
X, y = [], []
sequence_length = 60
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Reshape for LSTM input
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train-test split
X_train, X_test = X[:-30], X[-30:]
y_train, y_test = y[:-30], y[-30:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict and inverse transform
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual, label="Actual")
plt.plot(predictions, label="LSTM Forecast")
plt.title("LSTM Forecast vs Actual")
plt.legend()
plt.grid(True)
plt.show()


# In[44]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import pandas as pd

# Sample structure of df (simulate the presence of real stock data)
# You should replace this with loading your actual dataset
# Example:
# df = pd.read_csv('your_stock_data.csv', parse_dates=['Date'], index_col='Date')

# For demonstration, let's mock the 'df' structure
# Uncomment and modify below if using real data
# df = pd.read_csv("your_stock_data.csv", parse_dates=["Date"], index_col="Date")

# Placeholder for forecast results
forecast_arima = np.random.rand(30) * 100 + 150  # Dummy ARIMA forecast
sarima_forecast = np.random.rand(30) * 100 + 150  # Dummy SARIMA forecast
predictions = np.random.rand(30) * 100 + 150  # Dummy LSTM forecast

# Create mock test data
test_dates = pd.date_range(start="2024-01-01", periods=30)
test = pd.Series(np.random.rand(30) * 100 + 150, index=test_dates)

# Create mock Prophet forecast
forecast_prophet = pd.DataFrame({
    "ds": test_dates,
    "yhat": np.random.rand(30) * 100 + 150
})

# ✅ Align Prophet forecast to test dates
prophet_forecast_30 = forecast_prophet.set_index('ds').loc[test.index]

# ✅ Evaluate Models
arima_rmse = np.sqrt(mean_squared_error(test, forecast_arima))
arima_mae = mean_absolute_error(test, forecast_arima)

sarima_rmse = np.sqrt(mean_squared_error(test, sarima_forecast))
sarima_mae = mean_absolute_error(test, sarima_forecast)

prophet_rmse = np.sqrt(mean_squared_error(test, prophet_forecast_30['yhat']))
prophet_mae = mean_absolute_error(test, prophet_forecast_30['yhat'])

lstm_rmse = np.sqrt(mean_squared_error(test.values, predictions))
lstm_mae = mean_absolute_error(test.values, predictions)

# ✅ Print results
print("📊 Model Comparison (Last 30 Days Forecast)")
print(f"{'Model':<10} | {'RMSE':<10} | {'MAE':<10}")
print("-" * 36)
print(f"{'ARIMA':<10} | {arima_rmse:.2f}     | {arima_mae:.2f}")
print(f"{'SARIMA':<10} | {sarima_rmse:.2f}     | {sarima_mae:.2f}")
print(f"{'Prophet':<10} | {prophet_rmse:.2f}     | {prophet_mae:.2f}")
print(f"{'LSTM':<10} | {lstm_rmse:.2f}     | {lstm_mae:.2f}")

# ✅ Plot bar chart comparison
models = ['ARIMA', 'SARIMA', 'Prophet', 'LSTM']
rmses = [arima_rmse, sarima_rmse, prophet_rmse, lstm_rmse]
maes = [arima_mae, sarima_mae, prophet_mae, lstm_mae]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width / 2, rmses, width, label='RMSE', color='cornflowerblue')
plt.bar(x + width / 2, maes, width, label='MAE', color='salmon')
plt.xticks(x, models)
plt.title("📊 Error Comparison of Forecasting Models")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# In[45]:


get_ipython().system('pip install streamlit pandas matplotlib plotly')


# In[47]:


# Save this as app.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.title("📈 Stock Forecast Dashboard")

# ✅ Load your dataset
df = pd.read_csv("data/stock_data.csv", parse_dates=["timestamp"], index_col="timestamp")

# ✅ Show actual columns
st.sidebar.write("Columns in dataset:", df.columns.tolist())

# ✅ If 'close' is lowercase, we use it
col_name = "close" if "close" in df.columns else "Close"

# ✅ Plot actual data
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[col_name], mode='lines', name='Actual Close'))

# ✅ Load model forecasts (you must have these CSVs ready)
try:
    arima = pd.read_csv("forecasts/arima_forecast.csv", parse_dates=["timestamp"], index_col="timestamp")
    sarima = pd.read_csv("forecasts/sarima_forecast.csv", parse_dates=["timestamp"], index_col="timestamp")
    prophet = pd.read_csv("forecasts/prophet_forecast.csv", parse_dates=["timestamp"], index_col="timestamp")
    lstm = pd.read_csv("forecasts/lstm_forecast.csv", parse_dates=["timestamp"], index_col="timestamp")

    # Add forecast lines
    fig.add_trace(go.Scatter(x=arima.index, y=arima["forecast"], mode='lines+markers', name="ARIMA Forecast"))
    fig.add_trace(go.Scatter(x=sarima.index, y=sarima["forecast"], mode='lines+markers', name="SARIMA Forecast"))
    fig.add_trace(go.Scatter(x=prophet.index, y=prophet["yhat"], mode='lines+markers', name="Prophet Forecast"))
    fig.add_trace(go.Scatter(x=lstm.index, y=lstm["forecast"], mode='lines+markers', name="LSTM Forecast"))

except Exception as e:
    st.warning(f"⚠️ Could not load some forecast files: {e}")

# ✅ Customize plot
fig.update_layout(
    title="Forecast Comparison",
    xaxis_title="Date",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)


# In[ ]:




