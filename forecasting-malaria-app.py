import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Malaria Forecasting System", layout="wide")

st.title("Malaria Forecasting & Predictive Intelligence System")
st.markdown("Upload your malaria dataset and generate forecasts up to 2030 using Prophet.")

# Sidebar controls
st.sidebar.header("Forecast Controls")

forecast_year = st.sidebar.number_input(
    "Forecast Until Year",
    min_value=2025,
    max_value=2050,
    value=2030
)

seasonality = st.sidebar.checkbox("Enable Yearly Seasonality", value=True)

changepoint_scale = st.sidebar.slider(
    "Changepoint Prior Scale",
    min_value=0.001,
    max_value=0.5,
    value=0.05
)

train_ratio = st.sidebar.slider(
    "Train/Test Split Ratio",
    min_value=0.5,
    max_value=0.95,
    value=0.8
)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Attempt automatic column detection
    date_col = None
    target_col = None

    for col in df.columns:
        if "year" in col.lower() or "date" in col.lower():
            date_col = col
        if "case" in col.lower() or "malaria" in col.lower():
            target_col = col

    if date_col is None or target_col is None:
        st.error("Could not automatically detect date or target column. Please rename appropriately.")
        st.stop()

    df = df[[date_col, target_col]].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna()
    df = df.sort_values(by=date_col)

    df = df.rename(columns={date_col: "ds", target_col: "y"})

    st.subheader("Processed Data Summary")
    st.write(df.describe())
    st.write(f"Time Range: {df['ds'].min()} to {df['ds'].max()}")
    st.write(f"Total Observations: {len(df)}")

    # Train/Test Split
    split_index = int(len(df) * train_ratio)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    # Initialize Prophet




    # Forecast until chosen year
    last_year = df['ds'].dt.year.max()
    periods = forecast_year - last_year

    
    

    # Merge predictions for evaluation
        

    # Forecast Plot
    st.subheader("Forecast Visualization")

    # Actual vs Predicted


    # Components
    st.subheader("Trend & Seasonality Components")

    # Download Forecast
    st.subheader("Download Forecast Data")
    

else:

    st.info("Please upload a CSV file to begin forecasting.")




# malaria_forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import os

# -----------------------------
# App Title
# -----------------------------
st.title("Malaria Forecasting Dashboard 🦟")
st.write("Forecast malaria cases using historical data with Prophet or SARIMA models.")

# -----------------------------
# Upload Historical Data
# -----------------------------
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'ds' and 'y' columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # -----------------------------
    # Model Selection
    # -----------------------------
    model_choice = st.sidebar.selectbox("Select Forecasting Model", ["Prophet", "SARIMA"])

    # -----------------------------
    # Prophet Forecast
    # -----------------------------
    if model_choice == "Prophet":
        model_file = "malaria_prophet_model.pkl"

        if os.path.exists(model_file):
            model = joblib.load(model_file)
            st.success("Loaded pre-trained Prophet model!")
        else:
            st.success("Trained new Prophet model!")

        # Forecast Settings
        st.sidebar.header("Forecast Settings")
        periods = st.sidebar.number_input("Forecast periods (days)", min_value=1, max_value=365, value=30)

        st.subheader("Forecasted Data")

        # Visualizations
        st.subheader("Forecast Plot")

        st.subheader("Components Plot")
        st.pyplot(fig2)

    # -----------------------------
    # SARIMA Forecast
    # -----------------------------
    else:
        sarima_file = "malaria_sarima_model.pkl"

        # Allow user to set SARIMA parameters
        st.sidebar.header("SARIMA Parameters")
        p = st.sidebar.number_input("AR term (p)", min_value=0, max_value=5, value=1)
        d = st.sidebar.number_input("Difference term (d)", min_value=0, max_value=2, value=1)
        q = st.sidebar.number_input("MA term (q)", min_value=0, max_value=5, value=1)
        P = st.sidebar.number_input("Seasonal AR (P)", min_value=0, max_value=5, value=1)
        D = st.sidebar.number_input("Seasonal Difference (D)", min_value=0, max_value=2, value=1)
        Q = st.sidebar.number_input("Seasonal MA (Q)", min_value=0, max_value=5, value=1)
        s = st.sidebar.number_input("Seasonal Period (s)", min_value=1, max_value=365, value=12)
        forecast_steps = st.sidebar.number_input("Forecast periods", min_value=1, max_value=365, value=30)

        if os.path.exists(sarima_file):
            sarima_model = joblib.load(sarima_file)
            st.success("Loaded pre-trained SARIMA model!")
        else:
            sarima_model = SARIMAX(df['y'], order=(p,d,q), seasonal_order=(P,D,Q,s), enforce_stationarity=False, enforce_invertibility=False)
            sarima_model = sarima_model.fit(disp=False)
            joblib.dump(sarima_model, sarima_file)
            st.success("Trained new SARIMA model!")

        # Forecast
        sarima_forecast = sarima_model.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=df['ds'].iloc[-1], periods=forecast_steps+1, freq='D')[1:]
        sarima_df = pd.DataFrame({
            'ds': forecast_index,
            'yhat': sarima_forecast.predicted_mean,
            'yhat_lower': sarima_forecast.conf_int()['lower y'],
            'yhat_upper': sarima_forecast.conf_int()['upper y']
        })

        st.subheader("Forecasted Data")
        st.dataframe(sarima_df)

        # Plot
        st.subheader("SARIMA Forecast Plot")
        plt.figure(figsize=(10,5))
        plt.plot(pd.to_datetime(df['ds']), df['y'], label='Actual')
        plt.plot(sarima_df['ds'], sarima_df['yhat'], label='Forecast')
        plt.fill_between(sarima_df['ds'], sarima_df['yhat_lower'], sarima_df['yhat_upper'], color='pink', alpha=0.3)
        plt.xlabel("Date")
        plt.ylabel("Malaria Cases")
        plt.legend()
        st.pyplot(plt)

    # -----------------------------
    # Optional: Confusion/Error Metrics
    # -----------------------------
    if st.sidebar.checkbox("Compare with actuals (MSE)"):
        if 'y_actual' in df.columns:
            if model_choice == "Prophet":
                y_true = df['y_actual']
                y_pred = forecast['yhat'][:len(y_true)]
            else:
                y_true = df['y_actual']
                y_pred = sarima_df['yhat'][:len(y_true)]
            mse = mean_squared_error(y_true, y_pred)
            st.write(f"Mean Squared Error: {mse:.2f}")
        else:
            st.warning("CSV must include 'y_actual' column for comparison.")

else:
    st.info("Please upload a CSV file to get started.")















































