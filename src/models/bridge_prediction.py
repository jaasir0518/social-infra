"""
Bridge condition prediction model.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator

from .base_model import BaseInfrastructureModel
from utils.exceptions import ModelTrainingError


class BridgePredictor(BaseInfrastructureModel):
    """
    Model for predicting bridge condition and maintenance needs.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize bridge predictor.
        
        Args:
            model_params: Model hyperparameters
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params)
    
    def _create_model(self) -> BaseEstimator:
        """
        Create the Random Forest model for bridge prediction.
        
        Returns:
            Random Forest Regressor
        """
        return RandomForestRegressor(**self.model_params)
    
    def predict_maintenance_priority(
        self,
        X: pd.DataFrame,
        condition_threshold: float = 6.0
    ) -> pd.DataFrame:
        """
        Predict maintenance priority for bridges.
        
        Args:
            X: Features DataFrame
            condition_threshold: Threshold below which maintenance is needed
            
        Returns:
            DataFrame with predictions and priorities
        """
        try:
            # Get condition predictions
            condition_predictions = self.predict(X)
            
            # Calculate maintenance priority
            priorities = []
            for score in condition_predictions:
                if score < 5.0:
                    priority = 'Critical'
                elif score < condition_threshold:
                    priority = 'High'
                elif score < 7.5:
                    priority = 'Medium'
                else:
                    priority = 'Low'
                priorities.append(priority)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'predicted_condition': condition_predictions,
                'maintenance_needed': condition_predictions < condition_threshold,
                'maintenance_priority': priorities,
                'urgency_score': np.maximum(0, condition_threshold - condition_predictions)
            }, index=X.index)
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Maintenance priority prediction failed: {e}")
    
    def predict_deterioration_rate(
        self,
        X: pd.DataFrame,
        years_ahead: int = 5
    ) -> pd.DataFrame:
        """
        Predict bridge condition deterioration over time.
        
        Args:
            X: Features DataFrame
            years_ahead: Number of years to predict ahead
            
        Returns:
            DataFrame with deterioration predictions
        """
        try:
            current_condition = self.predict(X)
            
            # Simple deterioration model (can be made more sophisticated)
            # Assume deterioration rate depends on current condition and age
            if 'age' in X.columns:
                age_factor = (X['age'] / 50.0).clip(0.1, 2.0)  # Normalize age impact
            else:
                age_factor = pd.Series(1.0, index=X.index)
            
            # Base deterioration rate per year
            base_deterioration = 0.1
            annual_deterioration = base_deterioration * age_factor
            
            # Predict future conditions
            future_predictions = []
            for year in range(1, years_ahead + 1):
                future_condition = current_condition - (annual_deterioration * year)
                future_condition = np.maximum(future_condition, 0)  # Can't go below 0
                future_predictions.append(future_condition)
            
            # Create results DataFrame
            results = pd.DataFrame(
                np.array(future_predictions).T,
                columns=[f'condition_year_{i+1}' for i in range(years_ahead)],
                index=X.index
            )
            
            results['current_condition'] = current_condition
            results['deterioration_rate'] = annual_deterioration
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Deterioration prediction failed: {e}")