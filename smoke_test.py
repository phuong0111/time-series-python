"""
Smoke test: Verify data loading, model creation, training (3 epochs), and evaluation.
Tests LSTM-AE + MSE on SMD dataset.
"""
import os, sys, random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from src.config import AppConfig, SMDOptions, LSTMOptions
from src.const import DatasetType, ModelType, LossType
from src.data_loader.factory import DataLoaderFactory
from src.model.factory import ModelFactory
from src.loss.factory import LossStrategyFactory

def main():
    cfg = AppConfig()
    cfg.data.dataset_type = DatasetType.SMD
    cfg.data.data_path = 'data/SMD'
    cfg.data.window_size = 80
    cfg.data.smd = SMDOptions(entity_id='machine-1-1')
    cfg.data.use_cache = False

    cfg.model.model_type = ModelType.LSTM_AE
    cfg.model.lstm = LSTMOptions(lstm_units=[128], activation='tanh')
    cfg.model.epochs = 3  # Quick test only
    cfg.model.batch_size = 32

    cfg.loss.loss_type = LossType.MSE

    # 1. Load
    print('=== Step 1: Loading data ===')
    loader = DataLoaderFactory.get_loader(cfg.data)
    train_set, val_set, test_set = loader.get_data()
    X_train, _ = train_set
    X_val, _ = val_set if val_set else (None, None)
    X_test, y_test = test_set
    print(f'  X_train: {X_train.shape}')
    print(f'  X_val:   {X_val.shape if X_val is not None else None}')
    print(f'  X_test:  {X_test.shape}')
    print(f'  y_test:  {y_test.shape}, anomaly ratio: {y_test.mean():.4f}')

    # 2. Build
    print('\n=== Step 2: Building LSTM-AE ===')
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_wrapper = ModelFactory.get_model(cfg.model, input_shape)
    model_wrapper.model.summary()

    # 3. Train
    print('\n=== Step 3: Training (3 epochs) ===')
    strategy = LossStrategyFactory.get_strategy(cfg.loss)
    history, weights = strategy.execute(
        model_wrapper=model_wrapper, X_train=X_train, X_val=X_val
    )

    # 4. Evaluate
    print('\n=== Step 4: Evaluating ===')
    metrics = model_wrapper.evaluate(X_test, y_test, feature_weights=weights)
    
    print(f'\n{"="*50}')
    print(f'SMOKE TEST RESULT:')
    print(f'  F1:        {metrics["best_f1"]:.4f}')
    print(f'  AUC:       {metrics["roc_auc"]:.4f}')
    print(f'  Precision: {metrics["precision"]:.4f}')
    print(f'  Recall:    {metrics["recall"]:.4f}')
    print(f'{"="*50}')
    print('✅ SMOKE TEST PASSED!')

if __name__ == '__main__':
    main()
