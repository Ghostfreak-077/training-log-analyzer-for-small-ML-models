import numpy as np
import pandas as pd

from parser import parse_log


def analyze_training(df):
    if df.empty:
        raise ValueError("No training records found.")

    results = {}

    # Basic statistics
    results["total_epochs"] = len(df)

    if "loss" in df.columns:
        loss = df["loss"].dropna().to_numpy()
        if len(loss) > 0:
            results["average_training_loss"] = np.mean(loss)
            results["minimum_training_loss"] = np.min(loss)

    if "val_loss" in df.columns:
        val_loss = df["val_loss"].dropna()

        if not val_loss.empty:
            best_index = val_loss.idxmin()
            results["best_validation_loss"] = val_loss.loc[best_index]
            results["best_epoch"] = df.loc[best_index, "epoch"]

    if "accuracy" in df.columns:
        accuracy = df["accuracy"].dropna().to_numpy()

        if len(accuracy) > 0:
            results["average_accuracy"] = np.mean(accuracy)
            results["maximum_accuracy"] = np.max(accuracy)

    return results


if __name__ == "__main__":
    df = parse_log("data/sample_log.txt")
    summary = analyze_training(df)

    for metric, value in summary.items():
        print(f"{metric}: {value}")