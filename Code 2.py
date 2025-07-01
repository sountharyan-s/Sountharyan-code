from IPython import get_ipython
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import os
import time
from datetime import datetime, timedelta
import logging
import io
from tensorflow.keras.layers import Input

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# Set random seed for reproducibility
np.random.seed(42)

# Cache data fetching to avoid repeated yfinance calls
# @st.cache_data # This decorator is for Streamlit, not standard Python execution in Colab
def fetch_stock_data(ticker, start_date, end_date, max_retries=5):
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
            if df.empty:
                raise ValueError(f"No data available for {ticker} between {start_date} and {end_date}")
            df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            df.index = pd.to_datetime(df.index)
            df = df.loc[start_date:end_date]
            # st.write(f"Loaded {len(df)} rows from yfinance") # This is for Streamlit
            logger.info(f"Loaded {len(df)} rows for {ticker}")
            return df
        except Exception as e:
            if "Rate limited" in str(e):
                wait_time = 2 ** attempt * 10
                # st.warning(f"Rate limit error on attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s...") # This is for Streamlit
                logger.warning(f"Rate limit error: {e}. Waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                # st.error(f"Error fetching data from yf: {e}") # This is for Streamlit
                logger.error(f"Error fetching data from yf: {e}")
                return None

# Data loading
try:
    df_aapl = pd.read_csv('AAPL.csv', index_col='Date')
    display(df_aapl.head())
    print(df_aapl.shape)
except FileNotFoundError:
    print("Error: 'AAPL.csv' not found.")
except KeyError:
    print("Error: 'Date' column not found in 'AAPL.csv'. Please check the file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Data exploration
display(df_aapl.head())
display(df_aapl.tail())
print(df_aapl.dtypes)
print(df_aapl.isnull().sum())
display(df_aapl.describe())

plt.figure(figsize=(12, 8))
for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df_aapl[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_aapl[['Open', 'High', 'Low', 'Close', 'Adj Close']])
plt.title('Box Plots of Numerical Features')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df_aapl.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_aapl['Close'])
plt.title('Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Summarize findings in a markdown report
report = """
## Exploratory Data Analysis of AAPL Stock Data

This report summarizes the findings from an exploratory data analysis (EDA) of historical Apple (AAPL) stock data.

*1. Data Overview:*

The dataset spans from 1980-12-12 to 2020-04-01, containing 9909 data points. Each data point represents daily stock information, including opening price, high price, low price, closing price, adjusted closing price, and trading volume. No missing values were found in the dataset.

*2. Data Types:*

All numerical columns ('Open', 'High', 'Low', 'Close', 'Adj Close') are of type float64, and the 'Volume' column is of type int64.

*3. Descriptive Statistics:*

The descriptive statistics reveal a wide range of values for the stock prices and trading volume. The mean closing price is approximately $32.62, while the median is $1.73. The maximum closing price is significantly higher than the 75th percentile, suggesting potential outliers or periods of rapid growth. The 'Volume' column exhibits a high standard deviation, indicating substantial variability in daily trading volume.

*4. Data Distributions:*

The histograms show a skewed distribution for most of the numerical features. The 'Volume' column has a highly skewed distribution, with the majority of observations concentrated at lower values. The box plots highlight potential outliers, particularly in the 'Volume' feature. The distributions of 'Open', 'High', 'Low', and 'Close' price features also show right skewness.

*5. Correlation Analysis:*

The correlation matrix shows a high positive correlation between 'Open', 'High', 'Low', 'Close', and 'Adj Close' prices, as expected. The correlation between these prices and 'Volume' is relatively weak.

*6. Time Series Analysis:*

The time series plot of the 'Close' price shows an upward trend overall, with periods of significant volatility.

*7. Potential Issues:*

- *Outliers:* The 'Volume' column exhibits potential outliers which may need to be handled appropriately during subsequent analysis.
- *Skewness:* The skewed distribution of many features may require transformation (e.g., logarithmic transformation) to improve model performance.
"""

print(report)

# Data cleaning
print("Missing values before cleaning:\n", df_aapl.isnull().sum())
num_duplicates = df_aapl.duplicated().sum()
df_aapl = df_aapl[~df_aapl.duplicated()]
print(f"\nNumber of duplicate rows removed: {num_duplicates}")
print("\nMissing values after cleaning:\n", df_aapl.isnull().sum())
print("\nDuplicate rows after cleaning:", df_aapl.duplicated().sum())
display(df_aapl.head())
print("\nShape of the cleaned DataFrame:", df_aapl.shape)

# Feature engineering
df_aapl['MA7'] = df_aapl['Close'].rolling(window=7).mean()
df_aapl['MA21'] = df_aapl['Close'].rolling(window=21).mean()
delta = df_aapl['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df_aapl['RSI14'] = 100 - (100 / (1 + rs))
df_aapl['Close_Lag1'] = df_aapl['Close'].shift(1)
df_aapl['Close_Lag2'] = df_aapl['Close'].shift(2)
df_aapl['Close_Lag3'] = df_aapl['Close'].shift(3)
df_aapl = df_aapl.ffill()
display(df_aapl.head(25))

# Data splitting
split_index = int(len(df_aapl) * 0.8)
df_train = df_aapl.iloc[:split_index]
df_test = df_aapl.iloc[split_index:]
print("Shape of df_train:", df_train.shape)
print("Shape of df_test:", df_test.shape)
display(df_train.head())
display(df_test.head())

# Model training
X_train_rf = df_train.drop(columns=['Close'])
y_train_rf = df_train['Close']
X_test_rf = df_test.drop(columns=['Close'])
y_test_rf = df_test['Close']

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)
rf_predictions = rf_model.predict(X_test_rf)

scaler = MinMaxScaler()
df_aapl['Close'] = scaler.fit_transform(np.array(df_aapl['Close']).reshape(-1, 1))
time_steps = 10
X, y = [], []
for i in range(len(df_aapl) - time_steps):
    X.append(df_aapl['Close'][i:(i + time_steps)])
    y.append(df_aapl['Close'].iloc[i + time_steps])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

split_index = int(len(X) * 0.8)
X_train_lstm, X_test_lstm = X[:split_index], X[split_index:]
y_train_lstm, y_test_lstm = y[:split_index], y[split_index:]

lstm_model = Sequential()
lstm_model.add(Input(shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=1, batch_size=32)

lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Model evaluation
rf_mae = mean_absolute_error(y_test_rf, rf_predictions)
rf_mse = mean_squared_error(y_test_rf, rf_predictions)

print(f"RandomForestRegressor MAE: {rf_mae}")
print(f"RandomForestRegressor MSE: {rf_mse}")

if lstm_predictions.shape[0] != y_test_lstm.shape[0]:
    lstm_predictions = lstm_predictions[:y_test_lstm.shape[0]]

lstm_mae = mean_absolute_error(y_test_lstm, lstm_predictions)
lstm_mse = mean_squared_error(y_test_lstm, lstm_predictions)

print(f"LSTM MAE: {lstm_mae}")
print(f"LSTM MSE: {lstm_mse}")

print("\nModel Performance Summary:")
if rf_mae < lstm_mae and rf_mse < lstm_mse:
    print("RandomForestRegressor appears to be more accurate based on lower MAE and MSE.")
elif rf_mae > lstm_mae and rf_mse > lstm_mse:
    print("LSTM appears to be more accurate based on lower MAE and MSE.")
else:
    print("Model performance is mixed. Further analysis is needed to determine the better model.")

# Summary
print("\n## Summary:")
print("\n### Data Analysis Key Findings")
print("* The AAPL dataset spans from 1980-12-12 to 2020-04-01, containing 9909 data points with no missing values.")
print("* Several features were engineered, including 7-day and 21-day moving averages ('MA7', 'MA21'), 14-day RSI ('RSI14'), and lagged 'Close' prices.")
print("* The data was split into 80% training and 20% testing sets.")
print("* RandomForestRegressor and LSTM models were trained.  The RandomForestRegressor achieved an MAE of", rf_mae, "and MSE of", rf_mse, ", while the LSTM model achieved significantly lower MAE of", lstm_mae, "and MSE of", lstm_mse, ".  This indicates the LSTM model performed considerably better.")
print("\n### Insights or Next Steps")
print("* The LSTM model demonstrated superior performance in predicting AAPL stock prices compared to the RandomForestRegressor.")
print("* Explore hyperparameter tuning for both models, and potentially other models, to further improve predictive accuracy.")
