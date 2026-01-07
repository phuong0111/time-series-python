from abc import ABC, abstractmethod
from src.loss.calculators import WeightCalculator
from src.config import LossConfig
import logging

class TrainingStrategy(ABC):
    def __init__(self, config: LossConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def execute(self, model_wrapper, X_train, X_val=None):
        """
        Executes training.
        Args:
            model_wrapper: Instance of BaseAnomalyDetector
            X_train: Training data
            X_val: Validation data (Optional)
        Returns:
            (history, weights): Tuple of Keras history and learned feature weights (if any)
        """
        pass

class MSEStrategy(TrainingStrategy):
    def execute(self, model_wrapper, X_train, X_val=None):
        self.logger.info("Executing Standard MSE Training")
        
        # 1. Prepare Validation
        val_data = (X_val, X_val) if X_val is not None else None
        
        # 2. Compile & Train
        model_wrapper.compile_model(custom_weights=None) 
        history = model_wrapper.train(X_train, validation_data=val_data, loss_name=self.config.loss_type.value)
        
        return history, None

class RFWeightedStrategy(TrainingStrategy):
    def execute(self, model_wrapper, X_train, X_val=None):
        self.logger.info("Executing Random Forest Weighted Training (Unsupervised)")
        
        # Extract config
        cfg = self.config.rf_weighted
        
        # 1. Calculate Unsupervised Importance (Real vs Fake)
        # We use X_train to find which features define the "normal" structure
        weights = WeightCalculator.calculate_unsupervised_rf_importance(
            X_train, 
            n_estimators=cfg.n_estimators, 
            random_state=cfg.random_state
        )
        
        # 2. Prepare Validation
        val_data = (X_val, X_val) if X_val is not None else None

        # 3. Train with Weights
        model_wrapper.compile_model(custom_weights=weights)
        history = model_wrapper.train(X_train, validation_data=val_data, loss_name=self.config.loss_type.value)
        
        return history, weights

class FeatureScaledStrategy(TrainingStrategy):
    def execute(self, model_wrapper, X_train, X_val=None):
        self.logger.info("Executing Feature-Scaled Reconstruction Training")
        
        cfg = self.config.feature_scaled
        val_data = (X_val, X_val) if X_val is not None else None
        
        # 1. Pre-training (Standard MSE)
        self.logger.info(f"(A) Pre-training phase ({cfg.pretrain_epochs} epochs)...")
        
        # Temporarily reduce epochs for pre-training
        original_epochs = model_wrapper.config.epochs
        # If pretrain_epochs is small, we might not need validation/callbacks here, 
        # but passing them is safer if pretrain is long.
        
        model_wrapper.compile_model(custom_weights=None)
        # We manually fit here to avoid overwriting the main checkpoint or stopping too early
        model_wrapper.model.fit(
            X_train, X_train,
            epochs=cfg.pretrain_epochs,
            batch_size=model_wrapper.config.batch_size,
            verbose=1,
            shuffle=True
        )
        
        # 2. Calculate Weights (Inverse Reconstruction Error)
        self.logger.info("(B) Calculating Feature Weights...")
        weights = WeightCalculator.calculate_inverse_mse(
            model_wrapper.model, 
            X_train,
            epsilon=cfg.epsilon
        )
        
        # 3. Fine-tuning (Weighted MSE)
        self.logger.info("(C) Fine-tuning phase...")
        # Restore original epoch count
        # Note: The model continues from current weights (Transfer Learning)
        model_wrapper.compile_model(custom_weights=weights)
        history = model_wrapper.train(X_train, validation_data=val_data, loss_name=self.config.loss_type.value)
        
        return history, weights