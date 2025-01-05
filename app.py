import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------------------------------------
# 1. Define the LSTM training & prediction function
# -----------------------------------------------------------
def train_lstm_predict(
    data, 
    lat, 
    lon, 
    future_steps=4, 
    window_size=12, 
    epochs=20, 
    plot_results=True
):
    """
    Train an LSTM model on the time series of RAINFALL -> GROUNDWATER 
    for a particular (lat, lon). Then:
      1) Predict on the test set (in-sample predictions).
      2) Forecast 'future_steps' steps beyond the training data (out-of-sample).
    
    Args:
        data (pd.DataFrame): DataFrame with columns ['LAT', 'LON', 'TIME', 'RAINFALL', 'GROUNDWATER']
        lat (float): Latitude to filter data
        lon (float): Longitude to filter data
        future_steps (int): Number of future steps to forecast
        window_size (int): Number of time steps used as input for each prediction
        epochs (int): Number of training epochs for the LSTM
        plot_results (bool): If True, returns a matplotlib Figure of actual vs. predicted data

    Returns:
        dict: A dictionary with:
          - 'model': The trained LSTM model
          - 'scaler_X': The fitted MinMaxScaler for X
          - 'scaler_y': The fitted MinMaxScaler for y
          - 'predictions_test': Model predictions on the test set (inverted scale)
          - 'y_test': True test targets (inverted scale)
          - 'future_predictions': Multi-step future forecasts (inverted scale)
          - 'fig': (Optional) matplotlib Figure if plot_results=True
    """
    # 1. Filter and sort data
    group = data[(data['LAT'] == lat) & (data['LON'] == lon)].sort_values(by='TIME')

    # If no valid data, return None
    if group['RAINFALL'].isnull().all() or group['GROUNDWATER'].isnull().all():
        return {
            'error': f"No valid data for LAT: {lat}, LON: {lon}. Skipping..."
        }

    # 2. Prepare arrays for X and y
    X = group['RAINFALL'].fillna(0).values.reshape(-1, 1)
    y = group['GROUNDWATER'].fillna(0).values.reshape(-1, 1)

    if len(X) == 0 or len(y) == 0:
        return {
            'error': f"Insufficient data for LAT: {lat}, LON: {lon}. Skipping..."
        }

    # 3. Scale data (MinMaxScaler)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 4. Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - window_size):
        X_seq.append(X_scaled[i : i + window_size])
        # The label is the next time step after the window
        y_seq.append(y_scaled[i + window_size])  

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if X_seq.size == 0 or y_seq.size == 0:
        return {
            'error': f"Not enough sequences for LAT: {lat}, LON: {lon}. Skipping..."
        }

    # 5. Train/Test split
    split_index = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

    # 6. Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=epochs, verbose=0)  # verbose=0 for cleaner Streamlit output

    # 7. In-sample predictions (Test set)
    if len(X_test) > 0:
        predictions_test = model.predict(X_test)
        predictions_test_inv = scaler_y.inverse_transform(predictions_test)
        y_test_inv = scaler_y.inverse_transform(y_test)
    else:
        predictions_test_inv = None
        y_test_inv = None

    # 8. Future forecasting
    future_predictions_scaled = []
    future_input = X_seq[-1].copy()  # last window from the training set

    for _ in range(future_steps):
        pred_scaled = model.predict(future_input.reshape(1, window_size, 1))
        future_predictions_scaled.append(pred_scaled[0, 0])
        
        # Shift window: drop first value, add new predicted value
        future_input = np.append(future_input[1:], pred_scaled).reshape(window_size, 1)

    # Invert scaling for future predictions
    future_predictions = scaler_y.inverse_transform(
        np.array(future_predictions_scaled).reshape(-1, 1)
    )

    # 9. (Optional) Create Plot
    fig = None
    if plot_results and predictions_test_inv is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot historical actual
        ax.plot(range(len(y)), y, label='Actual (All Data)', c='blue')
        
        # Plot train/test boundary
        ax.axvline(split_index + window_size, color='gray', linestyle='--', label='Train/Test Split')

        # Plot test predictions
        test_indices = range(split_index + window_size, split_index + window_size + len(predictions_test_inv))
        ax.plot(test_indices, predictions_test_inv, label='Predicted (Test)', c='red')
        
        # Plot future predictions right after the last index of the dataset
        future_indices = range(len(y), len(y) + future_steps)
        ax.plot(future_indices, future_predictions, label='Forecast (Future)', c='green')
        
        ax.set_title(f"LSTM Prediction\nLAT: {lat}, LON: {lon}")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Groundwater Level")
        ax.legend()
        plt.tight_layout()

    # 10. Return results
    return {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'predictions_test': predictions_test_inv,
        'y_test': y_test_inv,
        'future_predictions': future_predictions,
        'fig': fig
    }

# -----------------------------------------------------------
# 2. Streamlit Application
# -----------------------------------------------------------
def main():
    st.title("LSTM Groundwater Forecasting App")

    # Read data once (use cache if data is large)
    @st.cache_data
    def load_data(csv_file="reshaped_groundwater_rainfall_data.csv"):
        return pd.read_csv(csv_file)
    
    data = load_data()

    # Sidebar user inputs
    st.sidebar.header("Model Parameters")
    lat = st.sidebar.number_input("Enter the latitude", value=26.88472222, format="%.8f")
    lon = st.sidebar.number_input("Enter the longitude", value=78.37916667, format="%.8f")
    future_steps = st.sidebar.number_input("Enter number of future steps", min_value=1, value=4)
    window_size = st.sidebar.number_input("Window Size", min_value=1, value=12)
    epochs = st.sidebar.number_input("Epochs", min_value=1, value=20)
    plot_results = st.sidebar.checkbox("Plot Results", value=True)

    if st.sidebar.button("Run Forecast"):
        with st.spinner("Training LSTM model..."):
            results = train_lstm_predict(
                data, 
                lat, 
                lon, 
                future_steps=future_steps, 
                window_size=window_size, 
                epochs=epochs, 
                plot_results=plot_results
            )
        
        # Check if any error
        if 'error' in results:
            st.error(results['error'])
        else:
            # Display future predictions
            future_predictions = results['future_predictions']
            if future_predictions is not None:
                st.write(f"**Future Groundwater Levels for LAT: {lat}, LON: {lon}**")
                future_vals = future_predictions.flatten()
                for i, val in enumerate(future_vals, start=1):
                    st.write(f"Future Step {i}: **{val:.2f}**")

            # Display Plot if generated
            if results['fig'] is not None:
                st.pyplot(results['fig'])
    else:
        st.write("Provide inputs in the sidebar and click **Run Forecast**.")

if __name__ == "__main__":
    main()
