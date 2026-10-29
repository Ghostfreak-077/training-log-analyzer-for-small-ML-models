import pandas as pd


def diagnose_training(df, overfit_threshold=2):
    """
    Detect potential training issues.

    overfit_threshold: Number of consecutive epochs
    with increasing validation loss required to flag overfitting.
    """
    diagnostics = []

    if df.empty:
        return ["No training data available."]

    if "loss" in df.columns and "val_loss" in df.columns:
        data = df.dropna(subset=["loss", "val_loss"])

        if len(data) >= 2:
            # Check whether validation loss keeps increasing
            val_loss = data["val_loss"]
            increases = val_loss.diff().iloc[1:] > 0

            consecutive_increases = 0
            max_consecutive_increases = 0

            for increased in increases:
                if increased:
                    consecutive_increases += 1
                    max_consecutive_increases = max(
                        max_consecutive_increases,
                        consecutive_increases
                    )
                else:
                    consecutive_increases = 0

            # Potential overfitting
            if (
                max_consecutive_increases >= overfit_threshold
                and data["loss"].iloc[-1] < data["loss"].iloc[0]
            ):
                diagnostics.append(
                    "Potential overfitting: training loss decreased "
                    "while validation loss increased consecutively."
                )

            # Check for unstable validation loss
            if len(val_loss) >= 3:
                fluctuations = val_loss.diff().abs().dropna()

                if fluctuations.max() > 0.5:
                    diagnostics.append(
                        "Large validation loss fluctuation detected."
                    )

    if "loss" in df.columns:
        loss = df["loss"].dropna()

        if len(loss) >= 3 and loss.iloc[-1] >= loss.iloc[0]:
            diagnostics.append(
                "Training loss did not improve overall."
            )

    if not diagnostics:
        diagnostics.append(
            "No issues detected by the current diagnostic rules."
        )

    return diagnostics


if __name__ == "__main__":
    from parser import parse_log

    df = parse_log("data/sample_log.txt")

    for message in diagnose_training(df):
        print("-", message)