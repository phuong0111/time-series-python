import sys
import os
import random
import pandas as pd
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

# === GLOBAL SEED FOR REPRODUCIBILITY ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from src.config import (
    AppConfig, DataConfig, ModelConfig, LossConfig, 
    DatasetType, ModelType, LossType, 
    SMDOptions, CICOptions, LSTMOptions, TCNOptions, TransformerOptions,
    FeatureScaledOptions, RFWeightedOptions, AdaptiveFeatureScaledOptions
)
from src.utils.logger import setup_logger
from src.data_loader.factory import DataLoaderFactory
from src.model.factory import ModelFactory
from src.loss.factory import LossStrategyFactory

# === SETUP LOGGER ===
Path("logs").mkdir(exist_ok=True)

log_config = AppConfig().logging
log_config.log_file = "logs/full_experiment.log"
setup_logger(log_config)
logger = logging.getLogger("ExperimentRunner")

def get_base_config():
    """Returns a fresh config object with defaults."""
    return AppConfig()

def run_single_experiment(dataset_type, model_type, loss_type):
    """
    Runs a single configuration (1 Dataset, 1 Model, 1 Loss).
    Returns a dictionary of metrics.
    """
    logger.info(f"STARTING EXPERIMENT: Dataset={dataset_type.value} | Model={model_type.value} | Loss={loss_type.value}")
    
    # 1. Initialize Config
    cfg = get_base_config()
    
    # --- CONFIGURE DATASET ---
    cfg.data.dataset_type = dataset_type
    cfg.data.window_size = 80  # Consistent window size
    
    if dataset_type == DatasetType.SMD:
        cfg.data.data_path = "data/SMD"
        cfg.data.smd = SMDOptions(entity_id="machine-1-1")
        cfg.model.dataset_name = cfg.data.smd.entity_id
    elif dataset_type == DatasetType.CIC:
        cfg.data.data_path = "data/CIC-DDoS2019/cicddos2019_dataset.csv"
        cfg.data.cic = CICOptions(
            label_column="Label",
            selected_columns=[
                "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
                "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
                "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Max",
                "Fwd IAT Total", "Fwd IAT Max", "FIN Flag Count", "SYN Flag Count", "RST Flag Count"
            ]
        )
        cfg.model.dataset_name = "CIC-DDoS2019"

    # --- CONFIGURE MODEL ---
    # All defaults come from config.py (batch=32, epochs=30, lr=0.001, dropout=0.1)
    cfg.model.model_type = model_type
    
    if model_type == ModelType.LSTM_AE:
        cfg.model.lstm = LSTMOptions(
            lstm_units=[128], 
            activation="tanh"
        )
    elif model_type == ModelType.TCN_AE:
        cfg.model.tcn = TCNOptions(
            kernel_size=3, 
            activation="relu",
            output_activation="linear"
        )
    elif model_type == ModelType.TRANSFORMER_AE:
        cfg.model.transformer = TransformerOptions(
            num_heads=4,
            key_dim=64,
            norm_epsilon=1e-6,
            activation="relu",
            output_activation="linear"
        )

    # --- CONFIGURE LOSS ---
    cfg.loss.loss_type = loss_type
    
    if loss_type == LossType.FEATURE_SCALED:
        cfg.loss.feature_scaled = FeatureScaledOptions(
            pretrain_epochs=10, 
            epsilon=1e-6
        )
    elif loss_type == LossType.RF_WEIGHTED:
        cfg.loss.rf_weighted = RFWeightedOptions(
            n_estimators=100,
            random_state=42
        )
    elif loss_type == LossType.ADAPTIVE_FEATURE_SCALED:
        cfg.loss.adaptive_feature_scaled = AdaptiveFeatureScaledOptions(
            pretrain_epochs=10,
            update_interval=5,
            epsilon=1e-6
        )

    # 2. Execution Pipeline
    try:
        # Load Data
        logger.info("Loading Data...")
        loader = DataLoaderFactory.get_loader(cfg.data)
        train_set, val_set, test_set = loader.get_data()
        
        X_train, _ = train_set
        X_val, _ = val_set if val_set else (None, None)
        X_test, y_test = test_set
        
        logger.info(f"Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Build Model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model_wrapper = ModelFactory.get_model(cfg.model, input_shape)
        
        # Train Strategy
        strategy = LossStrategyFactory.get_strategy(cfg.loss)
        
        logger.info("Executing Training Strategy...")
        history, train_weights = strategy.execute(
            model_wrapper=model_wrapper, 
            X_train=X_train, 
            X_val=X_val
        )
        
        # Evaluate
        logger.info("Evaluating...")
        metrics = model_wrapper.evaluate(X_test, y_test, feature_weights=train_weights)
        
        # Add metadata
        metrics['dataset'] = dataset_type.value
        metrics['model'] = model_type.value
        metrics['loss'] = loss_type.value
        metrics['status'] = 'Success'
        
        logger.info(f"RESULT: F1={metrics['best_f1']:.4f} | AUC={metrics['roc_auc']:.4f}")
        return metrics

    except Exception as e:
        logger.error(f"Experiment Failed: {e}", exc_info=True)
        return {
            'dataset': dataset_type.value,
            'model': model_type.value,
            'loss': loss_type.value,
            'status': 'Failed',
            'error': str(e),
            'best_f1': 0.0,
            'roc_auc': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

def main():
    # === EXPERIMENT GRID ===
    datasets = [DatasetType.SMD, DatasetType.CIC]
    models = [ModelType.LSTM_AE, ModelType.TCN_AE, ModelType.TRANSFORMER_AE]
    losses = [LossType.MSE, LossType.FEATURE_SCALED, LossType.RF_WEIGHTED, LossType.ADAPTIVE_FEATURE_SCALED]
    
    results = []
    total_exps = len(datasets) * len(models) * len(losses)
    
    print(f"--- STARTING FULL EVALUATION ---")
    print(f"Total Experiments: {total_exps}")
    print(f"Dataset: SMD (machine-1-1) | Window: 80 | Epochs: 30 | Batch: 32")
    print(f"Seed: {SEED}")
    
    count = 1
    for dataset in datasets:
        for model in models:
            for loss in losses:
                print(f"\n[{count}/{total_exps}] Running {dataset.value} - {model.value} - {loss.value}")
                
                res = run_single_experiment(dataset, model, loss)
                
                row = {
                    "Dataset": res['dataset'],
                    "Model": res['model'],
                    "Loss": res['loss'],
                    "F1": round(res.get('best_f1', 0), 4),
                    "Precision": round(res.get('precision', 0), 4),
                    "Recall": round(res.get('recall', 0), 4),
                    "AUC": round(res.get('roc_auc', 0), 4),
                    "Status": res['status']
                }
                
                results.append(row)
                count += 1
                
                # Incremental Save - Single JSON file
                with open("experiment_results.json", "w") as f:
                    json.dump(results, f, indent=4)
                
    print("\n--- ALL EXPERIMENTS COMPLETED ---")
    
    # Display table for final review
    if results:
        final_df = pd.DataFrame(results)
        print(final_df.to_markdown(index=False, tablefmt="grid"))

if __name__ == "__main__":
    main()