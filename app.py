import os
import tempfile

import streamlit as st

from parser import parse_log
from analyzer import analyze_training
from visualizer import plot_training_metrics
from diagnostics import diagnose_training

st.set_page_config(page_title="ML Training Log Analyzer")

st.title("ML Training Log Analyzer")
st.write("Upload a machine-learning training log to analyze its performance.")

uploaded_file = st.file_uploader("Upload training log", type=["txt", "log"])

if uploaded_file:
    os.makedirs("outputs", exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".txt", delete=False
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        df = parse_log(temp_path)

        if df.empty:
            st.error("No supported training records found.")
        else:
            st.subheader("Extracted Metrics")
            st.dataframe(df, use_container_width=True)

            st.subheader("Training Summary")
            st.json(analyze_training(df))

            st.subheader("Training Diagnostics")
            for message in diagnose_training(df):
                st.write(f"- {message}")

            st.subheader("Training Curves")

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(df["epoch"], df["loss"], marker="o", label="Training Loss")

            if "val_loss" in df.columns:
                ax.plot(
                    df["epoch"],
                    df["val_loss"],
                    marker="s",
                    label="Validation Loss",
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)
            plt.close(fig)

    except Exception as error:
        st.error(f"Analysis failed: {error}")

    finally:
        os.unlink(temp_path)