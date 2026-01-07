import sys
import pandas as pd
import logging
import numpy as np
from pathlib import Path

# Update imports to match your project structure
from src.config import (
    AppConfig, DataConfig, ModelConfig, LossConfig, 
    DatasetType, ModelType, LossType, 
    SMDOptions, CICOptions, LSTMOptions, TCNOptions, TransformerOptions,
    FeatureScaledOptions, RFWeightedOptions
)
from src.utils.logger import setup_logger
from src.data_loader.factory import DataLoaderFactory
from src.model.factory import ModelFactory
from src.loss.factory import LossStrategyFactory

# === SETUP LOGGER ===
# Ensure directory exists
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
    cfg.data.batch_size = 32
    cfg.data.window_size = 80  # Consistent window size
    
    if dataset_type == DatasetType.SMD:
        cfg.data.data_path = "data/SMD"
        cfg.data.smd = SMDOptions(entity_id="machine-1-1")
        
    elif dataset_type == DatasetType.CIC:
        cfg.data.data_path = "data/CIC-DDoS2019/cicddos2019_dataset.csv"
        cfg.data.cic = CICOptions(
            selected_columns=[
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd Packets Length Total', 'Bwd Packets Length Total',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max',
            'Fwd IAT Total', 'Fwd IAT Max', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count'
            ],
            label_column="Class"
        )

    # --- CONFIGURE MODEL ---
    cfg.model.model_type = model_type
    cfg.model.epochs = 50       # Enough for convergence with EarlyStopping
    cfg.model.learning_rate = 0.001
    cfg.model.dropout = 0.2     # Standard dropout for robust training
    cfg.model.latent_dim = 16   # Bottleneck size
    
    # Specific Optimizations for NEW Architectures
    if model_type == ModelType.LSTM_AE:
        # BI-DIRECTIONAL LSTM:
        # Since Bi-LSTM doubles the output dimension, we use [64, 32] 
        # which effectively becomes [128, 64] in the forward pass.
        cfg.model.lstm = LSTMOptions(
            lstm_units=[64, 32], 
            activation="tanh" # Tanh is more stable for deep LSTMs
        )

    elif model_type == ModelType.TCN_AE:
        # RESIDUAL TCN:
        # Residual connections allow us to go deeper without vanishing gradients.
        # We assume 4 levels of dilation [1, 2, 4, 8] for ~30 step receptive field per layer
        cfg.model.tcn = TCNOptions(
            nb_filters=[32, 64, 64, 32], # Symmetric filter count often helps AE
            kernel_size=3, 
            dilations=[1, 2, 4, 8], 
            activation="relu",
            output_activation="linear"
        )

    elif model_type == ModelType.TRANSFORMER_AE:
        # TRANSFORMER (With Positional Encoding):
        cfg.model.transformer = TransformerOptions(
            num_heads=4,
            key_dim=64,
            ff_units=[128, 64], # Larger FF network for capacity
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

    # 2. Execution Pipeline
    try:
        # Load Data
        logger.info("Loading Data...")
        loader = DataLoaderFactory.get_loader(cfg.data)
        
        # Unpack Data: (X_train, X_val, X_test)
        # Note: Your BaseLoader now returns 3 values (train, val, test)
        train_set, val_set, test_set = loader.get_data()
        
        X_train, _ = train_set
        X_val, _ = val_set if val_set else (None, None)
        X_test, y_test = test_set
        
        logger.info(f"Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Build Model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model_wrapper = ModelFactory.get_model(cfg.model, input_shape)
        
        # Train Strategy
        # The Strategy pattern handles:
        # 1. Calculating weights (if RF/Feature Scaled)
        # 2. Compiling the model
        # 3. Training the model
        strategy = LossStrategyFactory.get_strategy(cfg.loss)
        
        # Execute Strategy
        # Returns: History object and any learned feature weights
        logger.info("Executing Training Strategy...")
        history, train_weights = strategy.execute(
            model_wrapper=model_wrapper, 
            X_train=X_train, 
            X_val=X_val
        )
        
        # Evaluate
        # We pass feature_weights to apply them during anomaly scoring
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
    # === DEFINE THE GRID ===
    # Using lists to define the grid search space
    datasets = [DatasetType.SMD, DatasetType.CIC] 
    # Testing all updated architectures
    models = [ModelType.LSTM_AE, ModelType.TCN_AE, ModelType.TRANSFORMER_AE]
    # Testing baseline MSE vs your custom losses
    losses = [LossType.MSE, LossType.RF_WEIGHTED, LossType.FEATURE_SCALED] # Start with MSE to verify architectures first
    
    results = []
    total_exps = len(datasets) * len(models) * len(losses)
    
    print(f"--- STARTING FULL EVALUATION ---")
    print(f"Total Experiments: {total_exps}")
    
    count = 1
    for dataset in datasets:
        for model in models:
            for loss in losses:
                print(f"\n[{count}/{total_exps}] Running {dataset.value} - {model.value} - {loss.value}")
                
                # Run Experiment
                res = run_single_experiment(dataset, model, loss)
                
                # Format Result for CSV
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
                
                # Incremental Save
                pd.DataFrame(results).to_csv("experiment_results_partial.csv", index=False)
                
    # Final Save
    final_df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = ["Dataset", "Model", "Loss", "F1", "Precision", "Recall", "AUC", "Status"]
    final_df = final_df[cols]
    
    final_df.to_csv("experiment_results_final.csv", index=False)
    
    print("\n--- ALL EXPERIMENTS COMPLETED ---")
    print(final_df.to_markdown(index=False, tablefmt="grid"))

if __name__ == "__main__":
    main()