# Measuring Error: Example with Merlion

Merlion is an open-source Python library developed by Salesforce, designed for time series forecasting and anomaly detection. It simplifies the end-to-end workflow of time series analysis by integrating data preprocessing, model training, evaluation, and visualization into a single framework. Merlion supports:

- Statistical models (e.g., ARIMA)

- Machine learning approaches

- Deep learning models

Merlion expects time series data in Pandas DataFrame format with timestamps, making it compatible with standard Python data manipulation libraries.

## Why Use Merlion?

- **Unified Framework:** Combines preprocessing, modeling, evaluation, and visualization in one package.

- **Multiple Model Support:** Supports a variety of models, including statistical, machine learning, and deep learning approaches.

- **Built-in Evaluation Metrics:** Provides standardized evaluation metrics, such as sMAPE, for consistent model comparison.

- **Anomaly Detection:** Includes both supervised and unsupervised anomaly detection models.

# Example: Forecasting Energy Demand Using Merlion

The following example demonstrates how to use Merlion for time series forecasting with data from ERCOT on energy demand in Texas.

## Loading and Preprocessing Data

    from merlion.utils.time_series import TimeSeries

    # Load dataset
    url = "https://raw.githubusercontent.com/kylejones200/time_series/main/ercot_load_data.csv"
    df = pd.read_csv(url)

    # Convert time column to datetime format and set index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Resample to hourly frequency
    df = df.resample('H').mean()
    df['values'] = df['values'].interpolate()

    # Convert to Merlion TimeSeries format
    ts = TimeSeries.from_pd(df)
    print(ts)

## Choosing and Configuring Models

Merlion supports various forecasting models, including Prophet and ARIMA.

    from merlion.models.forecast.prophet import Prophet, ProphetConfig
    from merlion.models.forecast.arima import Arima, ArimaConfig

    # Initialize Prophet with optimized hyperparameters
    prophet_config = ProphetConfig(
        add_seasonality="auto",
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05  # Better trend detection
    )
    prophet_model = Prophet(prophet_config)

    # Initialize ARIMA (manually tuned order)
    arima_model = Arima(ArimaConfig(order=(2, 1, 2), target_seq_index=0))

## Splitting Data and Training Models

    # Split data into training and test sets
    train_ratio = 0.8  # 80% training, 20% testing
    split_idx = int(len(df) * train_ratio)
    train_data = TimeSeries.from_pd(df.iloc[:split_idx])
    test_data = TimeSeries.from_pd(df.iloc[split_idx:])

    # Train the models
    prophet_model.train(train_data)
    arima_model.train(train_data)

## Forecasting and Evaluating Model Performance

Merlion provides several metrics for measuring forecast accuracy. In this example, we use sMAPE (Symmetric Mean Absolute Percentage Error).

    from merlion.evaluate.forecast import ForecastMetric

    # Generate forecasts
    prophet_forecast, _ = prophet_model.forecast(test_data.time_stamps)
    arima_forecast, _ = arima_model.forecast(test_data.time_stamps)

    # Compute sMAPE
    prophet_smape = ForecastMetric.sMAPE.value(test_data, prophet_forecast)
    arima_smape = ForecastMetric.sMAPE.value(test_data, arima_forecast)

    print(f"Prophet sMAPE: {prophet_smape:.2f}")
    print(f"ARIMA sMAPE: {arima_smape:.2f}")

## Visualizing the Forecasts

Merlion includes built-in visualization tools, but Matplotlib provides more flexibility:

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.to_pd(), label="Actual")
    plt.plot(prophet_forecast.to_pd(), label="Prophet Forecast", linestyle="--")
    plt.plot(arima_forecast.to_pd(), label="ARIMA Forecast", linestyle="--")
    plt.legend()
    plt.title("Prophet vs ARIMA Forecasting")
    plt.show()

# Anomaly Detection with Merlion

Merlion supports both supervised and unsupervised anomaly detection. In this example, we use an Isolation Forest model.

    from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig

    # Initialize an Isolation Forest model
    config = IsolationForestConfig()
    anomaly_model = IsolationForest(config)

    # Train the model on the dataset
    anomaly_model.train(train_data)

    # Generate anomaly scores
    anomalies = anomaly_model.get_anomaly_label(test_data)
    scores = anomaly_model.get_anomaly_score(test_data)

    # Plot anomaly scores
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.to_pd(), label="Original Data")
    plt.plot(scores.to_pd(), label="Anomaly Scores", color="red", linestyle="--")
    plt.legend()
    plt.title("Anomaly Detection with Merlion")
    plt.show()

# Model Comparison

Merlion simplifies benchmarking multiple models using a consistent evaluation framework:

    # Compare models on performance metrics
    results = []
    for model, name in zip([arima_model, prophet_model], ["Merlion ARIMA", "Prophet"]):
        forecast, _ = model.forecast(test_data.time_stamps)
        smape = ForecastMetric.sMAPE.value(test_data, forecast)
        results.append({"Model": name, "sMAPE": smape})

    # Convert results to DataFrame
    comparison_df = pd.DataFrame(results)
    print(comparison_df)

# Performance Results

            Model           sMAPE
    0       Merlion ARIMA   6.04
    1       Prophet         25.01

- **Merlion ARIMA** outperforms Prophet in this example.

- **Prophet** has higher sMAPE, possibly due to difficulty in capturing fluctuations in the data.

Merlion simplifies time series forecasting and anomaly detection in Python by providing:

- **Unified Workflow:** Combines preprocessing, modeling, evaluation, and visualization.

- **Multiple Models:** Supports statistical, machine learning, and deep learning approaches.

- **Built-In Metrics:** Standardized evaluation metrics such as sMAPE for consistent comparison.

- **Anomaly Detection:** Both supervised and unsupervised methods are included.

## Insights from ERCOT Dataset

- **Default Forecaster** and **Merlion ARIMA** deliver the best results for the ERCOT dataset.

- **Prophet** performs better with proper tuning, especially with Mean-Variance Normalization.

- **Box-Cox Transform** significantly increases error, suggesting it is unsuitable for this dataset.

While Merlion provides a comprehensive framework for time series analysis, it has some limitations:

- Limited support for auto-tuning hyperparameters (e.g., ARIMA order).

- Visualization tools are less flexible compared to Matplotlib or Plotly.

Despite these limitations, Merlion remains a handy tool for rapid prototyping and benchmarking multiple time series models in Python.

## Key Takeaways

- Statistical models (e.g., ARIMA)
- Machine learning approaches
- Deep learning models
- **Unified Framework:** Combines preprocessing, modeling, evaluation, and visualization in one package.
