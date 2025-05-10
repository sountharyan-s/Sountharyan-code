import streamlit as st
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

# Cache data fetching to avoid repeated yfinance calls
@st.cache_data
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
            st.write(f"Loaded {len(df)} rows from yfinance")
            logger.info(f"Loaded {len(df)} rows for {ticker}")
            return df
        except Exception as e:
            if "Rate limited" in str(e):
                wait_time = 2 ** attempt * 10
                st.warning(f"Rate limit error on attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s...")
                logger.warning(f"Rate limit error: {e}. Waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                st.error(f"Error fetching data from yfinance: {e}")
                logger.error(f"yfinance error: {e}")
                break
    st.error("Failed to fetch data from yfinance. Please upload a CSV file with stock data.")
    return pd.DataFrame()

# Load and validate CSV data
def load_csv_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain columns: {', '.join(required_columns)}")
            return pd.DataFrame()
        
        # Convert Date to datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isna().any():
            st.error("Invalid date format in CSV. Use YYYY-MM-DD.")
            return pd.DataFrame()
        
        df.set_index('Date', inplace=True)
        
        # Validate numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[numeric_cols].isna().any().any():
            st.error("Non-numeric values found in numeric columns.")
            return pd.DataFrame()
        
        df = df.sort_index()
        st.write(f"Loaded {len(df)} rows from CSV")
        logger.info(f"Loaded {len(df)} rows from CSV")
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        logger.error(f"CSV loading error: {e}")
        return pd.DataFrame()

# Data Preprocessing
def preprocess_data(df):
    if df.empty:
        st.warning("Input DataFrame is empty")
        logger.warning("Empty DataFrame in preprocess_data")
        return df
    df = df.ffill().dropna()
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        z_scores = np.abs(zscore(df[numeric_cols]))
        df = df[(z_scores < 3).all(axis=1)]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index = df.index.to_period('D').to_timestamp()
    st.write(f"After preprocessing: {len(df)} rows")
    logger.info(f"After preprocessing: {len(df)} rows")
    return df

# Feature Engineering
def engineer_features(df):
    if df.empty:
        st.warning("Input DataFrame is empty for feature engineering")
        logger.warning("Empty DataFrame in engineer_features")
        return df
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_3'] = df['Close'].shift(3)
    df['Rolling_Mean_7'] = df['Close'].rolling(window=7).mean()
    df['Rolling_Std_7'] = df['Close'].rolling(window=7).std()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    df = df.dropna()
    st.write(f"After feature engineering: {len(df)} rows")
    logger.info(f"After feature engineering: {len(df)} rows")
    return df

def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Exploratory Data Analysis (EDA)
def perform_eda(df, save_path='outputs/eda_plots'):
    if df.empty:
        st.warning("Cannot perform EDA: DataFrame is empty")
        logger.warning("Empty DataFrame in perform_eda")
        return
    os.makedirs(save_path, exist_ok=True)
    
    # Closing Price Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'], label='Closing Price')
    ax.set_title('Stock Closing Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
    plt.savefig(f'{save_path}/closing_price.png')
    plt.close()
    
    # Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
    plt.savefig(f'{save_path}/correlation_heatmap.png')
    plt.close()
    
    # Daily Returns Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Close'].pct_change().dropna(), bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Daily Returns')
    ax.set_xlabel('Daily Return')
    st.pyplot(fig)
    plt.savefig(f'{save_path}/daily_returns.png')
    plt.close()

# Model Building and Evaluation
def train_arima_model(data, order=(5,1,0)):
    try:
        logger.info("Training ARIMA model")
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        logger.info("ARIMA model trained successfully")
        return model_fit
    except Exception as e:
        st.error(f"ARIMA training failed: {e}")
        logger.error(f"ARIMA training failed: {e}")
        return None

def prepare_lstm_data(data, look_back=20):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    if X.size == 0:
        return None, None, scaler
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_lstm_model(X_train, y_train, look_back=20):
    try:
        logger.info("Training LSTM model")
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        logger.info("LSTM model trained successfully")
        return model
    except Exception as e:
        st.error(f"LSTM training failed: {e}")
        logger.error(f"LSTM training failed: {e}")
        return None

def train_random_forest_model(X_train, y_train):
    try:
        logger.info("Training Random Forest model")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Random Forest model trained successfully")
        return model
    except Exception as e:
        st.error(f"Random Forest training failed: {e}")
        logger.error(f"Random Forest training failed: {e}")
        return None

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
    return mae, rmse, mape

# Forecasting Function
def forecast_future(model, data, steps, scaler=None, look_back=20, is_lstm=False):
    if model is None:
        return np.array([])
    if is_lstm:
        last_sequence = data[-look_back:].values.reshape(-1, 1)
        last_sequence = scaler.transform(last_sequence)
        future_preds = []
        current_sequence = last_sequence.copy()
        for _ in range(steps):
            current_sequence_reshaped = current_sequence.reshape(1, look_back, 1)
            next_pred = model.predict(current_sequence_reshaped, verbose=0)
            future_preds.append(next_pred[0, 0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0, 0]
        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        return future_preds.flatten()
    else:
        forecast = model.forecast(steps=steps)
        return forecast

# Streamlit App
def main():
    st.title("Stock Price Prediction Dashboard")
    st.write("Project: Cracking the Market Code with AI-Driven Stock Price Prediction")
    
    # User Inputs
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2025, 5, 9))
    forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=60, value=30)
    look_back = st.sidebar.slider("LSTM Look Back Period", min_value=5, max_value=50, value=20)
    
    # Data Source Selection
    st.sidebar.header("Data Source")
    use_yfinance = st.sidebar.checkbox("Use yfinance (default)", value=True)
    uploaded_file = None
    if not use_yfinance:
        st.sidebar.write("Upload a CSV file with columns: Date (YYYY-MM-DD), Open, High, Low, Close, Adj Close, Volume")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if st.sidebar.button("Run Analysis"):
        with st.spinner("Fetching and processing data..."):
            # Fetch Data
            df = pd.DataFrame()
            if use_yfinance:
                df = fetch_stock_data(ticker, start_date, end_date)
            if df.empty and uploaded_file is not None:
                st.write("yfinance failed or not selected. Loading data from CSV...")
                df = load_csv_data(uploaded_file)
            
            if df.empty:
                st.error("Exiting: No data available from yfinance or CSV. Please check your inputs or upload a valid CSV.")
                st.write("CSV should have columns: Date (YYYY-MM-DD), Open, High, Low, Close, Adj Close, Volume")
                return
            
            # Preprocess and Engineer Features
            df = preprocess_data(df)
            if df.empty:
                st.error("Exiting: No data available after preprocessing")
                return
            
            df = engineer_features(df)
            if df.empty:
                st.error("Exiting: No data available after feature engineering")
                return
            
            # EDA
            st.header("Exploratory Data Analysis")
            perform_eda(df)
            
            # Prepare Data for Modeling
            features = ['SMA_20', 'EMA_12', 'RSI', 'MACD', 'Signal_Line', 'Lag_1', 'Lag_3', 
                        'Rolling_Mean_7', 'Rolling_Std_7', 'Volume_Ratio']
            target = 'Close'
            
            X = df[features]
            y = df[target]
            if X.empty or y.empty:
                st.error("Exiting: Features or target data is empty")
                return
            
            if len(X) < 10:
                st.error("Exiting: Not enough data for train-test split")
                return
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            st.write(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Train Models
            with st.spinner("Training models..."):
                # Random Forest
                rf_model = train_random_forest_model(X_train, y_train)
                rf_pred = rf_model.predict(X_test) if rf_model else np.array([])
                rf_metrics = evaluate_model(y_test, rf_pred) if rf_pred.size > 0 else (float('inf'), float('inf'), float('inf'))
                
                # ARIMA
                arima_model = train_arima_model(y_train)
                arima_pred = arima_model.forecast(steps=len(y_test)) if arima_model else np.array([])
                arima_metrics = evaluate_model(y_test.values, arima_pred) if arima_pred.size > 0 else (float('inf'), float('inf'), float('inf'))
                
                # LSTM
                lstm_X, lstm_y, scaler = prepare_lstm_data(y, look_back)
                if lstm_X is None or lstm_y is None:
                    st.warning("Exiting: Insufficient data for LSTM model")
                    lstm_metrics = (float('inf'), float('inf'), float('inf'))
                    lstm_pred = np.array([])
                else:
                    lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test = train_test_split(
                        lstm_X, lstm_y, test_size=0.2, shuffle=False)
                    lstm_model = train_lstm_model(lstm_X_train, lstm_y_train, look_back)
                    if lstm_model:
                        lstm_pred = lstm_model.predict(lstm_X_test)
                        lstm_pred = scaler.inverse_transform(lstm_pred)
                        lstm_y_test = scaler.inverse_transform([lstm_y_test])
                        lstm_metrics = evaluate_model(lstm_y_test.T, lstm_pred)
                    else:
                        lstm_metrics = (float('inf'), float('inf'), float('inf'))
                        lstm_pred = np.array([])
            
            # Test Set Predictions Plot
            st.header("Test Set Predictions")
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(y_test.index, y_test, label='Actual')
            if rf_pred.size > 0:
                ax.plot(y_test.index, rf_pred, label='Random Forest')
            if arima_pred.size > 0:
                ax.plot(y_test.index, arima_pred, label='ARIMA')
            if lstm_pred.size > 0:
                ax.plot(y_test.index[-len(lstm_pred):], lstm_pred, label='LSTM')
            ax.set_title('Stock Price Predictions (Test Set)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)
            plt.savefig('outputs/predictions_test.png')
            plt.close()
            
            # Future Forecast
            with st.spinner("Generating future forecasts..."):
                future_dates = pd.date_range(start=end_date, periods=forecast_days + 1, freq='D')[1:]
                arima_future = forecast_future(arima_model, y, forecast_days)
                lstm_future = forecast_future(lstm_model, y, forecast_days, scaler, look_back, is_lstm=True) if lstm_X is not None and lstm_model else np.array([])
            
            # Future Forecast Plot
            st.header(f"Future Forecast (Next {forecast_days} Days)")
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(y.index[-60:], y[-60:], label='Historical Close')
            if arima_future.size > 0:
                ax.plot(future_dates, arima_future, label='ARIMA Forecast')
            if lstm_future.size > 0:
                ax.plot(future_dates, lstm_future, label='LSTM Forecast')
            ax.set_title(f'Stock Price Forecast (Next {forecast_days} Days)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)
            plt.savefig('outputs/forecast_future.png')
            plt.close()
            
            # Model Metrics
            st.header("Model Performance Metrics")
            metrics_df = pd.DataFrame({
                'Model': ['Random Forest', 'ARIMA', 'LSTM'],
                'MAE': [rf_metrics[0], arima_metrics[0], lstm_metrics[0]],
                'RMSE': [rf_metrics[1], arima_metrics[1], lstm_metrics[1]],
                'MAPE (%)': [rf_metrics[2], arima_metrics[2], lstm_metrics[2]]
            })
            st.dataframe(metrics_df)
            metrics_df.to_csv('outputs/model_metrics.csv')
            
            # Future Forecast Data
            st.header("Future Forecast Data")
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'ARIMA_Forecast': arima_future if arima_future.size > 0 else [np.nan] * forecast_days,
                'LSTM_Forecast': lstm_future if lstm_future.size > 0 else [np.nan] * forecast_days
            })
            st.dataframe(forecast_df)
            forecast_df.to_csv('outputs/future_forecasts.csv')

if __name__ == "__main__":
    main()
