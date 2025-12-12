# Modular Anomaly Detection for Time Series

A production-grade, modular framework for Time Series Anomaly Detection using Deep Learning. This project implements multiple architectures (LSTM-AE, TCN-AE, Transformer-AE) and advanced loss strategies (MSE, Random Forest Weighted, Feature-Scaled) in a unified, configurable pipeline.

## 🚀 Features

* **Multi-Model Support:**
    * **LSTM Autoencoder:** Classic sequence-to-sequence reconstruction.
    * **TCN Autoencoder:** Temporal Convolutional Network with dilated convolutions.
    * **Transformer Autoencoder:** Attention-based reconstruction for capturing long-range dependencies.
* **Advanced Loss Strategies:**
    * **Standard MSE:** Baseline reconstruction loss.
    * **Feature-Scaled Loss:** Dynamically weights features based on their reconstruction difficulty (Inverse MSE).
    * **RF-Weighted Loss:** Weights features based on their importance score from a Random Forest classifier.
* **Robust Data Pipeline:**
    * Supports **SMD (Server Machine Dataset)** and **CIC-DDoS2019**.
    * Automatic caching (Pickle) for fast re-runs.
    * Sliding window sequence generation.
* **Production Ready:**
    * **Configuration via `.env`** 
    * **Logging:** Centralized logging to file and console.
    * **Checkpoints:** Auto-saves best models.
    * **Evaluation:** Automatic F1-Score thresholding (Best F1 Strategy).

## 📂 Project Structure

```text
├── data/                  # Raw datasets (SMD, CIC)
├── cache/                 # Processed data cache (.pkl)
├── checkpoints/           # Saved models (.keras)
├── logs/                  # Execution logs
├── src/
│   ├── config.py          # Pydantic configuration & .env loading
│   ├── const.py           # Enums (ModelType, LossType, etc.)
│   ├── data_loader/       # Factory pattern for Data Loaders
│   ├── model/             # Factory pattern for Models (LSTM, TCN, Transformer)
│   ├── loss/              # Strategy pattern for Loss functions & Weights
│   └── utils/             # Logger & Helpers
├── main.py                # Entry point
└── .env                   # Configuration file (Git ignored)
```
