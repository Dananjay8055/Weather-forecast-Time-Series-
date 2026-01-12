🌦️ Weather Forecasting Model (Time Series)
📌 Project Overview

This project implements a time-series based weather forecasting system that predicts tomorrow’s maximum temperature using historical daily weather data.
The system uses NOAA Climate Data Online (CDO) API as a real-world data source and supports automated near real-time prediction through scheduled execution.

🎯 Objectives

Predict next-day temperature using historical data

Understand time-series concepts such as lag, rolling averages, and seasonality

Build an automated forecasting pipeline

Visualize model predictions using 2D and 3D graphs

Simulate a real-world, production-style workflow

🧠 Key Concepts Used

Time Series Forecasting

Feature Engineering (Lag & Rolling Mean)

Supervised Machine Learning

Linear Regression

Train/Test Split for Temporal Data

Automated Pipelines (Scheduler-based execution)

🛠️ Tools & Technologies

Python

Pandas

Matplotlib

Scikit-learn

NOAA Climate Data Online API

Git & GitHub

📂 Project Structure
Weather forecast/
│
├── data/
│   └── daily_temperature.csv
│
├── output/
│   ├── weather_forecasting_output.png
│   └── weather_forecasting_3d.png
│
├── fetch_noaa_data.py
├── train_model.py
├── run_pipeline.py
├── config.py
├── requirements.txt
└── README.md

🌐 Data Source

NOAA Climate Data Online (CDO)

Dataset: GHCND (Daily Summaries)

Variable used: TMAX (Daily Maximum Temperature)

Data is fetched programmatically using NOAA’s official REST API with token-based authentication.

⚙️ How the System Works
1️⃣ Data Ingestion

Fetches daily temperature data from NOAA using the /data endpoint

Updates dataset automatically with the latest available records

2️⃣ Feature Engineering

Lag feature: Previous day temperature

Rolling average: Mean temperature of last 7 days

3️⃣ Model Training

Linear Regression model trained on historical data

Time-based train/test split (no random shuffling)

4️⃣ Prediction

Predicts tomorrow’s temperature

Updates prediction whenever new data is available

5️⃣ Visualization

2D plot: Actual vs Predicted temperatures

3D plot:

X-axis → Time

Y-axis → Actual temperature

Z-axis → Predicted temperature

⏰ Automation (Near Real-Time Forecasting)

The project supports Level-3 automation using a scheduler:

run_pipeline.py executes:

Data fetching

Model training

Prediction generation

Can be scheduled to run daily using Windows Task Scheduler

Ensures predictions stay up to date without manual execution

▶️ How to Run the Project
Install dependencies
pip install -r requirements.txt

Fetch latest weather data
python fetch_noaa_data.py

Train model & generate predictions
python train_model.py

Run full automated pipeline
python run_pipeline.py

📊 Output

Console

Mean Absolute Error (MAE)

Predicted temperature for tomorrow

Saved Images

output/weather_forecasting_output.png

output/weather_forecasting_3d.png

🧪 Evaluation Metric

Mean Absolute Error (MAE)
Measures the average prediction error in degrees Celsius.

🚀 Future Enhancements

Use advanced models (Random Forest, XGBoost, LSTM)

Add more weather variables (humidity, rainfall)

Build a live dashboard using Streamlit

Store predictions in a database

Deploy as a web service

📚 Academic Relevance

This project demonstrates:

Practical application of time-series forecasting

Real-world data handling via APIs

Automation and reproducibility

Visualization for model interpretation

Suitable for:

Mini project

Final year project

Machine learning coursework

Data science portfolio

👤 Author

Dananjay VM
Weather Forecasting using Time Series & Machine Learning

⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to fork or extend it!
