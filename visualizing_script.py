from src import visualizer
import os

os.makedirs("outputs", exist_ok=True)

files = os.listdir("outputs")
log_files = [f for f in files if f.startswith("log_") and f.endswith(".txt")]

log_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

visualizer.plot_training_metrics(visualizer.parse_log(f"outputs/{log_files[-1]}"))