import matplotlib.pyplot as plt
import re
from src.parser import parse_log


def plot_training_metrics(df):
    if df.empty:
        print("No training data available.")
        return

    # Plot training and validation loss
    if "loss" in df.columns and "val_loss" in df.columns:
        plt.figure(figsize=(8, 5))

        plt.plot(df["epoch"], df["loss"], marker="o", label="Training Loss")
        plt.plot(df["epoch"], df["val_loss"], marker="s", label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("outputs/loss_curve.png")
        plt.show()

    # Plot accuracy
    if "accuracy" in df.columns:
        plt.figure(figsize=(8, 5))

        plt.plot(df["epoch"], df["accuracy"], marker="o", label="Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("outputs/accuracy_curve.png")
        plt.show()


if __name__ == "__main__":
    df = parse_log("outputs/log_1.txt")
    plot_training_metrics(df)