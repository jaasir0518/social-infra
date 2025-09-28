"""
Housing market and condition prediction model.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator

from .base_model import BaseInfrastructureModel
from utils.exceptions import ModelTrainingError


class HousingPredictor(BaseInfrastructureModel):
    """
    Model for predicting housing market trends and maintenance needs.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize housing predictor.
        
        Args:
            model_params: Model hyperparameters
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params)
    
    def _create_model(self) -> BaseEstimator:
        """
        Create the Gradient Boosting model for housing prediction.
        
        Returns:
            Gradient Boosting Regressor
        """
        return GradientBoostingRegressor(**self.model_params)
    
    def predict_market_trends(
        self,
        X: pd.DataFrame,
        months_ahead: int = 12
    ) -> pd.DataFrame:
        """
        Predict housing market trends.
        
        Args:
            X: Features DataFrame
            months_ahead: Number of months to predict ahead
            
        Returns:
            DataFrame with market trend predictions
        """
        try:
            # Get current predictions
            current_ratings = self.predict(X)
            
            # Simple trend model (can be enhanced with more sophisticated time series)
            # Assume market trends based on current conditions and property characteristics
            trend_factors = []
            
            for idx, row in X.iterrows():
                # Base trend factor
                trend = 0.0
                
                # Age factor - newer properties might appreciate faster
                if 'age' in X.columns:
                    age = row['age']
                    if age < 10:
                        trend += 0.02  # 2% annual appreciation
                    elif age < 25:
                        trend += 0.01  # 1% annual appreciation
                    else:
                        trend += 0.005  # 0.5% annual appreciation
                
                # Property type factor
                if 'property_type_encoded' in X.columns:
                    prop_type = row['property_type_encoded']
                    if prop_type == 0:  # Assume 0 is residential
                        trend += 0.01
                    elif prop_type == 1:  # Assume 1 is commercial
                        trend += 0.005
                
                trend_factors.append(trend)
            
            trend_factors = np.array(trend_factors)
            
            # Predict future values
            future_predictions = []
            for month in range(1, months_ahead + 1):
                future_rating = current_ratings * (1 + trend_factors * month / 12)
                future_predictions.append(future_rating)
            
            # Create results DataFrame
            results = pd.DataFrame(
                np.array(future_predictions).T,
                columns=[f'rating_month_{i+1}' for i in range(months_ahead)],
                index=X.index
            )
            
            results['current_rating'] = current_ratings
            results['annual_trend'] = trend_factors
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Market trend prediction failed: {e}")
    
    def predict_maintenance_cost(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict maintenance costs for housing properties.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with maintenance cost predictions
        """
        try:
            # Get condition ratings
            condition_ratings = self.predict(X)
            
            # Estimate maintenance costs based on condition and property characteristics
            maintenance_costs = []
            
            for idx, row in X.iterrows():
                base_cost = 1000  # Base annual maintenance cost
                
                # Condition factor - lower condition means higher maintenance
                condition = condition_ratings[idx] if idx < len(condition_ratings) else 7.0
                condition_factor = (10 - condition) / 2  # Scale inversely with condition
                
                # Age factor
                if 'age' in X.columns:
                    age = row['age']
                    age_factor = max(1.0, age / 20)  # Older properties cost more to maintain
                else:
                    age_factor = 1.0
                
                # Size factor (if available)
                if 'assessed_value' in X.columns:
                    value = row['assessed_value']
                    size_factor = max(1.0, value / 200000)  # Higher value = more maintenance
                else:
                    size_factor = 1.0
                
                total_cost = base_cost * condition_factor * age_factor * size_factor
                maintenance_costs.append(total_cost)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'condition_rating': condition_ratings,
                'annual_maintenance_cost': maintenance_costs,
                'maintenance_per_rating_point': np.array(maintenance_costs) / np.maximum(condition_ratings, 1)
            }, index=X.index)
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Maintenance cost prediction failed: {e}")