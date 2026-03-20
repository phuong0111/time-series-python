import numpy as np
import sys
from pathlib import Path

# Path setup
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import AppConfig
from src.data_loader.factory import DataLoaderFactory
from src.model.factory import ModelFactory
from src.loss.factory import LossStrategyFactory
from src.utils.logger import setup_logger

def run_workflow():
    # 1. Config (Same as before)
    app_config = AppConfig()
    setup_logger(config=app_config.logging)

    # 2. Load Data
    loader = DataLoaderFactory.get_loader(app_config.data)
    train_set, val_set, test_set = loader.get_data()
    X_train, _ = train_set
    X_val, _ = val_set if val_set else (None, None)
    X_test, y_test = test_set

    # 3. Build Model
    model_wrapper = ModelFactory.get_model(app_config.model, input_shape=(X_train.shape[1], X_train.shape[2]))

    # 4. EXECUTE STRATEGY (The Clean Part)
    # ---------------------------------------------------------
    strategy = LossStrategyFactory.get_strategy(app_config.loss)
    
    # The strategy handles pre-training, weight calc, and compilation internally
    history, trained_weights = strategy.execute(model_wrapper, X_train, X_val=X_val)
    # ---------------------------------------------------------

    # 5. Evaluate
    metrics = model_wrapper.evaluate(
        X_test, 
        y_test, 
        feature_weights=trained_weights
    )

if __name__ == "__main__":
    run_workflow()