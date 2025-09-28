"""
Ensemble model for combining multiple infrastructure predictions.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression

from .base_model import BaseInfrastructureModel
from utils.exceptions import ModelTrainingError, ModelPredictionError


class EnsembleModel(BaseInfrastructureModel):
    """
    Ensemble model that combines predictions from multiple infrastructure models.
    """
    
    def __init__(
        self,
        models: Optional[Dict[str, BaseEstimator]] = None,
        method: str = 'voting',
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: Dictionary of trained models
            method: Ensemble method ('voting', 'stacking', 'averaging')
            weights: Weights for voting/averaging methods
        """
        super().__init__()
        self.models = models or {}
        self.method = method
        self.weights = weights
        self.meta_model = None
        
    def _create_model(self) -> BaseEstimator:
        """
        Create the ensemble model based on specified method.
        
        Returns:
            Ensemble estimator
        """
        if not self.models:
            raise ValueError("No models provided for ensemble")
        
        if self.method == 'voting':
            # Use VotingRegressor
            estimators = [(name, model) for name, model in self.models.items()]
            return VotingRegressor(
                estimators=estimators,
                weights=self.weights
            )
        
        elif self.method == 'stacking':
            # Use meta-learning approach
            self.meta_model = LinearRegression()
            return StackingEnsemble(self.models, self.meta_model)
        
        elif self.method == 'averaging':
            # Simple averaging ensemble
            return AveragingEnsemble(self.models, self.weights)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def add_model(self, name: str, model: BaseEstimator) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name: Name of the model
            model: Trained model to add
        """
        self.models[name] = model
        self.is_trained = False  # Need to retrain ensemble
    
    def remove_model(self, name: str) -> None:
        """
        Remove a model from the ensemble.
        
        Args:
            name: Name of the model to remove
        """
        if name in self.models:
            del self.models[name]
            self.is_trained = False  # Need to retrain ensemble
    
    def get_model_weights(self) -> Optional[Dict[str, float]]:
        """
        Get the weights assigned to each model.
        
        Returns:
            Dictionary of model weights
        """
        if self.weights and len(self.weights) == len(self.models):
            return dict(zip(self.models.keys(), self.weights))
        else:
            return None
    
    def predict_with_individual_models(
        self,
        X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Dictionary of predictions from each model
        """
        try:
            individual_predictions = {}
            
            for name, model in self.models.items():
                if hasattr(model, 'predict'):
                    predictions = model.predict(X)
                    individual_predictions[name] = predictions
            
            return individual_predictions
            
        except Exception as e:
            raise ModelPredictionError(f"Individual model predictions failed: {e}")
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions with confidence intervals based on model agreement.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with predictions and confidence metrics
        """
        try:
            # Get individual model predictions
            individual_preds = self.predict_with_individual_models(X)
            
            if not individual_preds:
                raise ValueError("No individual predictions available")
            
            # Get ensemble prediction
            ensemble_pred = self.predict(X)
            
            # Calculate confidence metrics
            pred_array = np.array(list(individual_preds.values())).T
            pred_std = np.std(pred_array, axis=1)
            pred_range = np.ptp(pred_array, axis=1)  # peak-to-peak range
            
            # Create confidence score (inverse of standard deviation)
            confidence = 1.0 / (1.0 + pred_std)
            
            results = pd.DataFrame({
                'ensemble_prediction': ensemble_pred,
                'prediction_std': pred_std,
                'prediction_range': pred_range,
                'confidence_score': confidence,
                'model_agreement': 1.0 - (pred_std / np.maximum(np.abs(ensemble_pred), 1.0))
            }, index=X.index)
            
            # Add individual model predictions
            for name, preds in individual_preds.items():
                results[f'{name}_prediction'] = preds
            
            return results
            
        except Exception as e:
            raise ModelPredictionError(f"Confidence prediction failed: {e}")


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble implementation.
    """
    
    def __init__(self, models: Dict[str, BaseEstimator], meta_model: BaseEstimator):
        self.models = models
        self.meta_model = meta_model
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the stacking ensemble.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Self
        """
        # Get predictions from base models (use cross-validation in practice)
        base_predictions = []
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                # Assume models are already fitted
                pred = model.predict(X)
                base_predictions.append(pred)
            else:
                # Fit model if not already fitted
                model.fit(X, y)
                pred = model.predict(X)
                base_predictions.append(pred)
        
        # Stack predictions as features for meta-model
        stacked_features = np.column_stack(base_predictions)
        
        # Train meta-model
        self.meta_model.fit(stacked_features, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions using stacking ensemble.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get base model predictions
        base_predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_features = np.column_stack(base_predictions)
        
        # Get meta-model prediction
        return self.meta_model.predict(stacked_features)


class AveragingEnsemble(BaseEstimator, RegressorMixin):
    """
    Simple averaging ensemble implementation.
    """
    
    def __init__(self, models: Dict[str, BaseEstimator], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the averaging ensemble (models assumed to be pre-fitted).
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Self
        """
        # For averaging ensemble, we assume base models are already fitted
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions using weighted averaging.
        
        Args:
            X: Features
            
        Returns:
            Averaged predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.weights:
            # Weighted average
            weights = np.array(self.weights[:len(predictions)])
            weights = weights / weights.sum()  # Normalize weights
            return np.average(predictions, axis=0, weights=weights)
        else:
            # Simple average
            return np.mean(predictions, axis=0)