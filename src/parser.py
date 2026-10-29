import re
import pandas as pd


def parse_log(file_path):
    records = []

    # Match epoch number
    epoch_pattern = r"Epoch\s+(\d+)/\d+"

    # Match metric values
    metric_patterns = {
        "loss": r"loss:\s*([\d.]+)",
        "val_loss": r"val_loss:\s*([\d.]+)",
        "accuracy": r"accuracy:\s*([\d.]+)"
    }

    with open(file_path, "r") as file:
        for line in file:
            epoch_match = re.search(epoch_pattern, line)

            if not epoch_match:
                continue

            record = {
                "epoch": int(epoch_match.group(1))
            }

            for metric, pattern in metric_patterns.items():
                match = re.search(pattern, line)
                record[metric] = (
                    float(match.group(1)) if match else None
                )

            records.append(record)

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = parse_log("data/sample_log.txt")
    print(df)
