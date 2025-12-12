from abc import ABC, abstractmethod
from src.loss.calculators import WeightCalculator
from src.config import LossConfig

class TrainingStrategy(ABC):
    def __init__(self, config: LossConfig):
        self.config = config

    @abstractmethod
    def execute(self, model_wrapper, X_train, X_test=None, y_test=None):
        """
        Orchestrates the training process.
        Args:
            model_wrapper: Instance of BaseAnomalyDetector (LSTM/TCN/etc)
            X_train: Training data
            X_test: Test data (needed for RF)
            y_test: Labels (needed for RF)
        """
        pass

class MSEStrategy(TrainingStrategy):
    def execute(self, model_wrapper, X_train, X_test=None, y_test=None):
        print(">>> [Strategy] Executing Standard MSE Training")
        # 1. Compile with default MSE
        model_wrapper.compile_model(custom_weights=None) 
        # 2. Train normally
        model_wrapper.train(X_train)

class RFWeightedStrategy(TrainingStrategy):
    def execute(self, model_wrapper, X_train, X_test=None, y_test=None):
        print(">>> [Strategy] Executing Random Forest Weighted Training")
        
        if X_test is None or y_test is None:
            raise ValueError("RF Strategy requires X_test and y_test for importance calculation.")

        # 1. Calculate Weights
        weights = WeightCalculator.calculate_rf_importance(X_test, y_test)
        
        # 2. Re-compile model with these weights
        model_wrapper.compile_model(custom_weights=weights)
        
        # 3. Train
        model_wrapper.train(X_train)

class FeatureScaledStrategy(TrainingStrategy):
    def execute(self, model_wrapper, X_train, X_test=None, y_test=None):
        print(">>> [Strategy] Executing Feature-Scaled Reconstruction Training")
        
        # 1. Pre-training (Fast, unweighted)
        print("   (A) Pre-training phase (MSE)...")
        original_epochs = model_wrapper.config.epochs
        
        # Temporarily reduce epochs for pre-training
        model_wrapper.config.epochs = self.config.pretrain_epochs
        model_wrapper.compile_model(custom_weights=None) # MSE
        model_wrapper.train(X_train)
        
        # 2. Calculate Weights (Inverse MSE)
        print("   (B) Calculating Feature Weights...")
        weights = WeightCalculator.calculate_inverse_mse(model_wrapper.model, X_train)
        
        # 3. Fine-tuning (Weighted)
        print("   (C) Fine-tuning phase (Weighted)...")
        # Restore original epoch count
        model_wrapper.config.epochs = original_epochs
        model_wrapper.compile_model(custom_weights=weights)
        model_wrapper.train(X_train)