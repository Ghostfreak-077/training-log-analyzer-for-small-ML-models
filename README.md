# training-log-analyzer-for-small-ML-models

A simple tool for analyzing training logs of small machine learning models.

## Usage

1. Run the training script to generate log files in the `outputs` directory.

```
   python src/model/train.py
```
2. Execute the visualization script to analyze the training logs:

   ```bash
   python visualizing_script.py
   ```

This will create plots of the training and validation loss, as well as the model accuracy, saved in the `outputs` directory.

