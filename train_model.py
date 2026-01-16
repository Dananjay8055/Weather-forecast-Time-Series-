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

        # --- 3D visualization (optional) ---
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            import numpy as np

            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(111, projection='3d')

            time_index = np.arange(len(dates))

            # Actual vs Predicted in 3D
            ax.plot(time_index, y_test.values, predictions, color="#00E5FF", linewidth=2.5, label="Actual vs Predicted")
            ax.scatter(time_index, y_test.values, predictions, color="#FF6F00", s=8)

            ax.set_title("3D Weather Forecasting Model (Time Series Prediction)", fontsize=16, fontweight='bold')
            ax.set_xlabel("Time index")
            ax.set_ylabel("Actual TMAX (°C)")
            ax.set_zlabel("Predicted TMAX (°C)")

            out3d = os.path.join(output_dir, "weather_forecasting_3d.png")
            plt.savefig(out3d, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"3D plot saved to: {out3d}")
        except Exception as e:
            print(f"Skipping 3D plot generation: {e}")

    # Save trained model
    if save_model:
        model_path = os.path.join(model_dir, "linear_regression.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

    # -----------------------------
    # Save latest prediction for frontend
    # -----------------------------
    try:
        # Build features for tomorrow: use last observed TMAX as lag_1 and
        # the most recent 7-day mean as rolling_7
        last_tmax = df["TMAX"].iloc[-1]
        last_7 = df["TMAX"].iloc[-7:].mean()
        tomorrow_X = pd.DataFrame([[last_tmax, last_7]], columns=["lag_1", "rolling_7"])
        tomorrow_prediction = model.predict(tomorrow_X)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "latest_prediction.txt"), "w") as f:
            f.write(f"{tomorrow_prediction[0]:.2f}")

        print(f"Latest prediction saved to: {os.path.join(output_dir, 'latest_prediction.txt')}")
    except Exception as e:
        print(f"Could not save latest prediction: {e}")


def main():
    train_and_save()


if __name__ == "__main__":
    main()
