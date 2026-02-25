import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    model = Prophet(
        yearly_seasonality=seasonality,
        changepoint_prior_scale=changepoint_scale
    )

    model.fit(train)

    # Forecast until chosen year
    last_year = df['ds'].dt.year.max()
    periods = forecast_year - last_year

    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)

    # Merge predictions for evaluation
    forecast_test = forecast[['ds', 'yhat']].merge(test, on='ds', how='inner')

    if not forecast_test.empty:
        mae = mean_absolute_error(forecast_test['y'], forecast_test['yhat'])
        rmse = np.sqrt(mean_squared_error(forecast_test['y'], forecast_test['yhat']))
        mape = np.mean(np.abs((forecast_test['y'] - forecast_test['yhat']) / forecast_test['y'])) * 100

        st.subheader("Model Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAPE (%)", f"{mape:.2f}")

    # Forecast Plot
    st.subheader("Forecast Visualization")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        mode='lines',
        name='Historical Data'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines',
        name='Forecast'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='lightgrey',
        name='Upper Confidence'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='lightgrey',
        name='Lower Confidence'
    ))

    fig.update_layout(
        title="Malaria Forecast Until {}".format(forecast_year),
        xaxis_title="Year",
        yaxis_title="Malaria Cases",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted
    if not forecast_test.empty:
        st.subheader("Actual vs Predicted Comparison")

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=forecast_test['ds'],
            y=forecast_test['y'],
            mode='lines+markers',
            name='Actual'
        ))

        fig2.add_trace(go.Scatter(
            x=forecast_test['ds'],
            y=forecast_test['yhat'],
            mode='lines+markers',
            name='Predicted'
        ))

        fig2.update_layout(
            title="Actual vs Predicted (Test Data)",
            xaxis_title="Year",
            yaxis_title="Malaria Cases"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Components
    st.subheader("Trend & Seasonality Components")
    components_fig = model.plot_components(forecast)
    st.pyplot(components_fig)

    # Download Forecast
    st.subheader("Download Forecast Data")
    forecast_download = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    st.download_button(
        label="Download Forecast CSV",
        data=forecast_download.to_csv(index=False),
        file_name="malaria_forecast.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to begin forecasting.")