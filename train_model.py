import os
from typing import Optional

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def train_and_save(
    data_path: str = "data/daily_temperature.csv",
    output_dir: str = "output",
    model_dir: str = "models",
    save_plot: bool = True,
    save_model: bool = True,
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if "DATE" not in df.columns or "TMAX" not in df.columns:
        raise RuntimeError("Input CSV must contain 'DATE' and 'TMAX' columns")

    df["DATE"] = pd.to_datetime(df["DATE"])

    # Feature engineering
    df["lag_1"] = df["TMAX"].shift(1)
    df["rolling_7"] = df["TMAX"].rolling(7).mean()
    df = df.dropna()

    if len(df) < 10:
        raise ValueError("Not enough data after feature engineering to train the model")

    X = df[["lag_1", "rolling_7"]]
    y = df["TMAX"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    print("Mean Absolute Error:", round(mae, 2))

    # Prepare output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save 2D plot (time series)
    if save_plot:
        dates = df["DATE"].iloc[split:]
        plt.style.use("dark_background")
        plt.figure(figsize=(12, 5))
        plt.plot(dates, y_test.values, label="Actual", color="#00E5FF", linewidth=2)
        plt.plot(dates, predictions, label="Predicted", color="#FF6F00", linewidth=2)
        plt.fill_between(dates, y_test.values, predictions, color="purple", alpha=0.12)
        plt.xlabel("Date")
        plt.ylabel("TMAX")
        plt.legend()
        plot_path = os.path.join(output_dir, "weather_forecasting_output.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_path}")

    # Save trained model
    if save_model:
        model_path = os.path.join(model_dir, "linear_regression.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")


def main():
    train_and_save()


if __name__ == "__main__":
    main()
