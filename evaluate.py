import os
import sys
import argparse
import logging
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import AppConfig, DatasetType, ModelType, LossType
from src.data_loader.factory import DataLoaderFactory
from src.model.factory import ModelFactory
from src.utils.logger import setup_logger

def setup_eval_logger():
    Path("logs").mkdir(exist_ok=True)
    cfg = AppConfig().logging
    cfg.log_file = "logs/evaluation.log"
    setup_logger(cfg)
    return logging.getLogger("Evaluator")

def evaluate_model(checkpoint_path: str, dataset_name: str, dataset_type_str: str, model_type_str: str):
    logger = setup_eval_logger()
    logger.info(f"--- Starting Evaluation ---")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset Name: {dataset_name} | Type: {dataset_type_str}")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        return

    # 1. Setup Config
    cfg = AppConfig()
    cfg.data.dataset_type = DatasetType(dataset_type_str)
    cfg.model.model_type = ModelType(model_type_str)
    
    # Configure dataset specific settings
    if cfg.data.dataset_type == DatasetType.SMD:
        cfg.data.data_path = "data/SMD"
        cfg.data.smd.entity_id = dataset_name
    elif cfg.data.dataset_type == DatasetType.CIC:
        cfg.data.data_path = "data/CIC-DDoS2019/cicddos2019_dataset.csv"
        cfg.data.cic.label_column = "Label" # Add logic here if needed

    # 2. Load Data
    logger.info("Loading Data...")
    try:
        loader = DataLoaderFactory.get_loader(cfg.data)
        _, _, test_set = loader.get_data()
        X_test, y_test = test_set
        logger.info(f"Test Data Loaded. Shape x: {X_test.shape}, y: {y_test.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        return

    # 3. Load Model
    logger.info("Loading Model Checkpoint...")
    input_shape = (X_test.shape[1], X_test.shape[2])
    model_wrapper = ModelFactory.get_model(cfg.model, input_shape)
    
    try:
        model_wrapper.load(checkpoint_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}", exc_info=True)
        return

    # 4. Evaluate
    # Note: We evaluate without feature_weights here as they are baked into training.
    # If the strategy was Adaptive, the model learned those weights in its kernels. 
    # (Inverse-MSE weights are only used during loss calculation in training).
    logger.info("Evaluating Model on Test Data...")
    try:
        metrics = model_wrapper.evaluate(X_test, y_test)
        
        print("\n" + "="*40)
        print("          EVALUATION RESULTS")
        print("="*40)
        print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
        print(f"Best F1:       {metrics['best_f1']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"Best Thresh:   {metrics['best_threshold']:.6f}")
        print("="*40 + "\n")
        
        # Save results to a simple json inside results/ folder
        os.makedirs("results", exist_ok=True)
        res_file = os.path.join("results", f"eval_results_{model_type_str}_{dataset_name}.json")
        with open(res_file, "w") as f:
            output = {k: v for k, v in metrics.items() if k != 'scores'} # exclude raw scores array from json
            json.dump(output, f, indent=4)
        logger.info(f"Results saved to {res_file}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained anomaly detection model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .keras model checkpoint")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset entity (e.g., machine-1-1)")
    parser.add_argument("--dataset-type", type=str, default="SMD", choices=["SMD", "CIC"], help="Type of dataset (SMD or CIC)")
    parser.add_argument("--model-type", type=str, default="LSTM_AE", choices=["LSTM_AE", "TCN_AE", "TRANSFORMER_AE"], help="Model architecture type")
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset_name,
        dataset_type_str=args.dataset_type,
        model_type_str=args.model_type
    )
