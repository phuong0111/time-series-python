from abc import ABC, abstractmethod
from src.loss.calculators import WeightCalculator
from src.config import LossConfig
import logging
import numpy as np

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
        
        model_wrapper.compile_model(custom_weights=None)
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
        model_wrapper.compile_model(custom_weights=weights)
        history = model_wrapper.train(X_train, validation_data=val_data, loss_name=self.config.loss_type.value)
        
        return history, weights


class AdaptiveFeatureScaledStrategy(TrainingStrategy):
    """
    Adaptive Feature-Scaled Strategy: recalculates feature weights every
    `update_interval` epochs so the model progressively refines which
    features matter most.

    Workflow:
        1. Pre-train with standard MSE for `pretrain_epochs`.
        2. Calculate initial feature weights from reconstruction error.
        3. Train for `update_interval` epochs with weighted loss.
        4. Recalculate weights and repeat step 3 until total epochs exhausted.
    """

    def execute(self, model_wrapper, X_train, X_val=None):
        self.logger.info("Executing Adaptive Feature-Scaled Reconstruction Training")

        cfg = self.config.adaptive_feature_scaled
        val_data = (X_val, X_val) if X_val is not None else None
        total_epochs = model_wrapper.config.epochs

        # ------------------------------------------------------------------
        # Phase A: Pre-training (Standard MSE)
        # ------------------------------------------------------------------
        self.logger.info(f"(A) Pre-training phase ({cfg.pretrain_epochs} epochs)...")
        model_wrapper.compile_model(custom_weights=None)
        model_wrapper.model.fit(
            X_train, X_train,
            epochs=cfg.pretrain_epochs,
            batch_size=model_wrapper.config.batch_size,
            verbose=1,
            shuffle=True,
        )

        # ------------------------------------------------------------------
        # Phase B–C: Iterative weight update loop
        # ------------------------------------------------------------------
        remaining_epochs = total_epochs
        n_rounds = max(1, remaining_epochs // cfg.update_interval)
        weight_history = []       # store weights from each round
        last_history = None       # keep the history of the final round
        weights = None

        for round_idx in range(n_rounds):
            # --- (B) Calculate / recalculate weights ---
            self.logger.info(
                f"(B) Round {round_idx + 1}/{n_rounds} — "
                f"Calculating Feature Weights..."
            )
            weights = WeightCalculator.calculate_inverse_mse(
                model_wrapper.model,
                X_train,
                epsilon=cfg.epsilon,
            )
            weight_history.append(weights.copy())

            # --- (C) Fine-tune for `update_interval` epochs ---
            epochs_this_round = (
                remaining_epochs - (n_rounds - 1 - round_idx) * cfg.update_interval
                if round_idx == n_rounds - 1
                else cfg.update_interval
            )
            self.logger.info(
                f"(C) Round {round_idx + 1}/{n_rounds} — "
                f"Fine-tuning for {epochs_this_round} epochs..."
            )
            model_wrapper.compile_model(custom_weights=weights)

            # Use model.fit directly so we don't reset callbacks / checkpoints
            # every round. The final round uses model_wrapper.train for the
            # proper checkpoint save.
            if round_idx < n_rounds - 1:
                model_wrapper.model.fit(
                    X_train, X_train,
                    epochs=epochs_this_round,
                    batch_size=model_wrapper.config.batch_size,
                    validation_data=val_data,
                    verbose=2,
                    shuffle=True,
                )
            else:
                # Last round — use full train() to get checkpointing
                original_epochs = model_wrapper.config.epochs
                model_wrapper.config.epochs = epochs_this_round
                last_history = model_wrapper.train(
                    X_train,
                    validation_data=val_data,
                    loss_name=self.config.loss_type.value,
                )
                model_wrapper.config.epochs = original_epochs

        # Log weight evolution summary
        if len(weight_history) > 1:
            delta = np.abs(weight_history[-1] - weight_history[0])
            self.logger.info(
                f"Weight drift (first → last): "
                f"mean={np.mean(delta):.4f}, max={np.max(delta):.4f}"
            )

        return last_history, weights