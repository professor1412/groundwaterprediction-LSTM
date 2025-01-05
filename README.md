LSTM Groundwater Forecasting App

This is a Streamlit web application for forecasting groundwater levels using an LSTM (Long Short-Term Memory) model. The application takes rainfall data as input and predicts groundwater levels for specific geographic locations (latitude and longitude). Additionally, it provides multi-step future forecasts beyond the training data.
Features

    Interactive UI: Enter latitude, longitude, window size, number of future steps, and epochs through the sidebar.
    LSTM Model Training: The app trains an LSTM model on the selected location’s data and predicts groundwater levels.
    Future Forecasting: Predicts future groundwater levels for a specified number of steps beyond the available data.
    Plot Results: Visualizes actual vs. predicted groundwater levels along with future forecasts.
    Error Handling: Provides informative error messages if data is insufficient for the selected location.

Requirements

Make sure you have Python 3.8 or higher installed, and then install the required packages using:

pip install -r requirements.txt

How to Run

    Clone the repository:

git clone https://github.com/professor1412/groundwaterprediction-LSTM.git
cd groundwaterprediction-LSTM

Create a virtual environment and activate it:

python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate

Install the dependencies:

pip install -r requirements.txt

Run the Streamlit app:

    streamlit run app.py

    Open your browser and navigate to http://localhost:8501.

File Descriptions

    app.py: The main Streamlit app that handles user interaction, input processing, and LSTM forecasting.
    lstm.ipynb: Jupyter notebook for exploratory data analysis and model development.
    reshaped_groundwater_rainfall_data.csv: The dataset containing historical rainfall and groundwater level data.
    requirements.txt: A list of required Python packages.
    .venv/: Virtual environment directory (excluded using .gitignore).

How It Works

    Data Input: The app reads the dataset and filters the data by the selected latitude and longitude.
    Preprocessing: The rainfall and groundwater level data are scaled and transformed into sequences for LSTM input.
    LSTM Model: A sequential LSTM model is trained using the selected parameters (window size, epochs).
    Prediction: The model predicts groundwater levels on the test set and forecasts future steps beyond the training data.
    Visualization: The app generates a plot showing actual vs. predicted values, including future forecasts.

Example Usage

    Enter the following sample inputs in the sidebar:
        Latitude: 26.88472222
        Longitude: 78.37916667
        Future Steps: 4
        Window Size: 12
        Epochs: 20
    Click Run Forecast.
    View the predicted future groundwater levels and plot in the main section.

License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.
Contact

For any questions or suggestions, feel free to contact professor1412.

Would you like any additional sections added, such as acknowledgments or further improvements?
