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

def evaluate_model(checkpoint_path: str, dataset_name: str, dataset_type_str: str, model_type_str: str) -> dict:
    logger = setup_eval_logger()
    logger.info(f"--- Starting Evaluation ---")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset Name: {dataset_name} | Type: {dataset_type_str}")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        return None

    # 1. Setup Config
    cfg = AppConfig(
        data={"dataset_type": dataset_type_str, "window_size": 80},
        model={"model_type": model_type_str, "dataset_name": dataset_name}
    )
    
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
        return None

    # 3. Load Model
    logger.info("Loading Model Checkpoint...")
    input_shape = (X_test.shape[1], X_test.shape[2])
    model_wrapper = ModelFactory.get_model(cfg.model, input_shape)
    
    try:
        model_wrapper.load(checkpoint_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}", exc_info=True)
        return None

    # 4. Evaluate
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
        
        output = {k: float(v) for k, v in metrics.items() if k != 'scores'} # exclude raw scores array from json
        
        # Save individual results to a simple json inside results/ folder
        os.makedirs("results", exist_ok=True)
        res_file = os.path.join("results", f"eval_results_{model_type_str}_{dataset_name}.json")
        with open(res_file, "w") as f:
            json.dump(output, f, indent=4)
        logger.info(f"Results saved to {res_file}")
        
        return output

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return None

def evaluate_all():
    logger = setup_eval_logger()
    logger.info("--- STARTING BATCH EVALUATION ---")
    
    datasets = [DatasetType.SMD, DatasetType.CIC]
    models = [ModelType.LSTM_AE, ModelType.TCN_AE, ModelType.TRANSFORMER_AE]
    losses = [LossType.MSE, LossType.FEATURE_SCALED, LossType.RF_WEIGHTED, LossType.ADAPTIVE_FEATURE_SCALED]
    
    all_results = []
    
    for dataset in datasets:
        dataset_name = "machine-1-1" if dataset == DatasetType.SMD else "CIC-DDoS2019"
        for model in models:
            for loss in losses:
                checkpoint_path = f"checkpoints/{dataset_name}/{model.value}_{loss.value}_best.keras"
                
                if os.path.exists(checkpoint_path):
                    print(f"\nEvaluating: {dataset.value} | {model.value} | {loss.value}")
                    metrics = evaluate_model(checkpoint_path, dataset_name, dataset.value, model.value)
                    
                    if metrics:
                        all_results.append({
                            "Dataset": dataset.value,
                            "Dataset_Name": dataset_name,
                            "Model": model.value,
                            "Loss": loss.value,
                            "F1": metrics.get("best_f1", 0.0),
                            "Precision": metrics.get("precision", 0.0),
                            "Recall": metrics.get("recall", 0.0),
                            "AUC": metrics.get("roc_auc", 0.0),
                            "Best_Threshold": metrics.get("best_threshold", 0.0)
                        })
                else:
                    logger.warning(f"Skipping {dataset.value} {model.value} {loss.value} - Checkpoint not found: {checkpoint_path}")

    # Save comprehensive results
    os.makedirs("results", exist_ok=True)
    final_res_file = os.path.join("results", "all_evaluations_summary.json")
    with open(final_res_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nBatch evaluation complete. Summary saved to {final_res_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained anomaly detection model.")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints found in the checkpoints directory")
    parser.add_argument("--checkpoint", type=str, help="Path to the saved .keras model checkpoint")
    parser.add_argument("--dataset-name", type=str, help="Name of the dataset entity (e.g., machine-1-1)")
    parser.add_argument("--dataset-type", type=str, choices=["SMD", "CIC"], help="Type of dataset (SMD or CIC)")
    parser.add_argument("--model-type", type=str, choices=["LSTM_AE", "TCN_AE", "TRANSFORMER_AE"], help="Model architecture type")
    
    args = parser.parse_args()
    
    if args.all:
        evaluate_all()
    elif args.checkpoint and args.dataset_name and args.dataset_type and args.model_type:
        evaluate_model(
            checkpoint_path=args.checkpoint,
            dataset_name=args.dataset_name,
            dataset_type_str=args.dataset_type,
            model_type_str=args.model_type
        )
    else:
        parser.print_help()
        print("\nError: You must provide either --all OR specify --checkpoint, --dataset-name, --dataset-type, and --model-type.")
