# tests/test_loss_workflow.py

import numpy as np
import sys
from pathlib import Path

# Path setup
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import AppConfig, DataConfig, ModelConfig, LossConfig, LossType, ModelType, DatasetType, SMDOptions
from src.data_loader.factory import DataLoaderFactory
from src.model.factory import ModelFactory
from src.loss.factory import LossStrategyFactory

def run_workflow():
    # 1. Config (Same as before)
    app_config = AppConfig(
        data=DataConfig(dataset_type=DatasetType.SMD, data_path="data/SMD"),
        model=ModelConfig(model_type=ModelType.LSTM_AE),
        loss=LossConfig(loss_type=LossType.FEATURE_SCALED) # <--- CHANGE THIS TO TEST
    )

    # 2. Load Data
    loader = DataLoaderFactory.get_loader(app_config.data)
    (X_train, _), (X_test, y_test) = loader.get_data()

    # 3. Build Model
    model_wrapper = ModelFactory.get_model(app_config.model, input_shape=(X_train.shape[1], X_train.shape[2]))

    # 4. EXECUTE STRATEGY (The Clean Part)
    # ---------------------------------------------------------
    strategy = LossStrategyFactory.get_strategy(app_config.loss)
    
    # The strategy handles pre-training, weight calc, and compilation internally
    strategy.execute(model_wrapper, X_train, X_test, y_test)
    # ---------------------------------------------------------

    # 5. Evaluate
    mse_scores = model_wrapper.get_anomaly_score(X_test)
    print(f"Mean Anomaly Score: {np.mean(mse_scores)}")

if __name__ == "__main__":
    run_workflow()