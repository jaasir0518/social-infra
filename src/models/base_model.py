"""
Base model class for infrastructure prediction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from utils.exceptions import ModelTrainingError, ModelPredictionError
from config.logging_config import LoggingMixin


class BaseInfrastructureModel(ABC, LoggingMixin):
    """
    Abstract base class for infrastructure prediction models.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            model_params: Model hyperparameters
        """
        self.model_params = model_params or {}
        self.model: Optional[BaseEstimator] = None
        self.is_trained = False
        self.feature_names_ = None
        self.target_name_ = None
        
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """
        Create the underlying ML model.
        
        Returns:
            Scikit-learn compatible model
        """
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseInfrastructureModel':
        """
        Train the model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Self for method chaining
        """
        try:
            self.logger.info(f"Training {self.__class__.__name__} model...")
            
            # Store feature and target names
            self.feature_names_ = list(X.columns)
            self.target_name_ = y.name
            
            # Create and train model
            self.model = self._create_model()
            self.model.fit(X, y)
            
            self.is_trained = True
            self.logger.info("Model training completed")
            
            return self
            
        except Exception as e:
            raise ModelTrainingError(f"Model training failed: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        try:
            if not self.is_trained or self.model is None:
                raise ModelPredictionError("Model must be trained before making predictions")
            
            return self.model.predict(X)
            
        except Exception as e:
            raise ModelPredictionError(f"Prediction failed: {e}")
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get prediction probabilities (if supported).
        
        Args:
            X: Features DataFrame
            
        Returns:
            Probability array or None if not supported
        """
        try:
            if not self.is_trained or self.model is None:
                raise ModelPredictionError("Model must be trained before making predictions")
            
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Probability prediction failed: {e}")
            return None
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Get model score.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Model score
        """
        try:
            if not self.is_trained or self.model is None:
                raise ModelPredictionError("Model must be trained before scoring")
            
            return self.model.score(X, y)
            
        except Exception as e:
            raise ModelPredictionError(f"Scoring failed: {e}")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation scores
        """
        try:
            if self.model is None:
                self.model = self._create_model()
            
            scores = cross_val_score(self.model, X, y, cv=cv)
            
            return {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {'mean_score': 0.0, 'std_score': 0.0, 'scores': []}
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance Series or None
        """
        try:
            if not self.is_trained or self.model is None:
                return None
            
            if hasattr(self.model, 'feature_importances_'):
                return pd.Series(
                    self.model.feature_importances_,
                    index=self.feature_names_
                ).sort_values(ascending=False)
            elif hasattr(self.model, 'coef_'):
                return pd.Series(
                    np.abs(self.model.coef_),
                    index=self.feature_names_
                ).sort_values(ascending=False)
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Feature importance extraction failed: {e}")
            return None
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Model parameters dictionary
        """
        if self.model is not None:
            return self.model.get_params()
        else:
            return self.model_params
    
    def set_params(self, **params) -> 'BaseInfrastructureModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        self.model_params.update(params)
        
        if self.model is not None:
            self.model.set_params(**params)
        
        return self