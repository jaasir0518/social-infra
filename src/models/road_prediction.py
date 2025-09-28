"""
Road condition and traffic prediction model.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import BaseEstimator

from .base_model import BaseInfrastructureModel
from utils.exceptions import ModelTrainingError


class RoadPredictor(BaseInfrastructureModel):
    """
    Model for predicting road condition and traffic patterns.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize road predictor.
        
        Args:
            model_params: Model hyperparameters
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        if model_params:
            default_params.update(model_params)
        
        super().__init__(default_params)
    
    def _create_model(self) -> BaseEstimator:
        """
        Create the Extra Trees model for road prediction.
        
        Returns:
            Extra Trees Regressor
        """
        return ExtraTreesRegressor(**self.model_params)
    
    def predict_pavement_life(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict remaining pavement life for roads.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with pavement life predictions
        """
        try:
            # Get current condition scores
            current_conditions = self.predict(X)
            
            # Estimate remaining pavement life
            remaining_life = []
            maintenance_timing = []
            
            for idx, row in X.iterrows():
                condition = current_conditions[idx] if idx < len(current_conditions) else 7.0
                
                # Base pavement life estimation
                if condition >= 8.0:
                    years_remaining = np.random.uniform(8, 12)
                    timing = 'Long-term'
                elif condition >= 7.0:
                    years_remaining = np.random.uniform(5, 8)
                    timing = 'Medium-term'
                elif condition >= 6.0:
                    years_remaining = np.random.uniform(2, 5)
                    timing = 'Short-term'
                else:
                    years_remaining = np.random.uniform(0, 2)
                    timing = 'Immediate'
                
                # Adjust for traffic volume if available
                if 'traffic_volume' in X.columns:
                    traffic = row['traffic_volume']
                    if traffic > 30000:  # High traffic
                        years_remaining *= 0.7
                    elif traffic > 15000:  # Medium traffic
                        years_remaining *= 0.85
                
                # Adjust for age if available
                if 'age' in X.columns:
                    age = row['age']
                    if age > 20:
                        years_remaining *= 0.8
                    elif age > 15:
                        years_remaining *= 0.9
                
                remaining_life.append(max(0, years_remaining))
                maintenance_timing.append(timing)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'current_condition': current_conditions,
                'remaining_pavement_life_years': remaining_life,
                'maintenance_timing': maintenance_timing,
                'rehabilitation_year': pd.Timestamp.now().year + np.array(remaining_life)
            }, index=X.index)
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Pavement life prediction failed: {e}")
    
    def predict_traffic_impact(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict traffic impact on road conditions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with traffic impact analysis
        """
        try:
            # Get current condition scores
            current_conditions = self.predict(X)
            
            # Calculate traffic impact factors
            traffic_impact = []
            congestion_risk = []
            
            for idx, row in X.iterrows():
                impact_score = 0.0
                
                # Traffic volume impact
                if 'traffic_volume' in X.columns:
                    volume = row['traffic_volume']
                    if volume > 40000:
                        impact_score += 3.0  # Very high impact
                        congestion = 'High'
                    elif volume > 25000:
                        impact_score += 2.0  # High impact
                        congestion = 'Medium-High'
                    elif volume > 15000:
                        impact_score += 1.0  # Medium impact
                        congestion = 'Medium'
                    else:
                        impact_score += 0.5  # Low impact
                        congestion = 'Low'
                else:
                    congestion = 'Unknown'
                
                # Lane capacity impact
                if 'lanes' in X.columns:
                    lanes = row['lanes']
                    if 'traffic_volume' in X.columns:
                        volume_per_lane = row['traffic_volume'] / lanes
                        if volume_per_lane > 10000:
                            impact_score += 1.5
                        elif volume_per_lane > 7500:
                            impact_score += 1.0
                        elif volume_per_lane > 5000:
                            impact_score += 0.5
                
                traffic_impact.append(impact_score)
                congestion_risk.append(congestion)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'current_condition': current_conditions,
                'traffic_impact_score': traffic_impact,
                'congestion_risk': congestion_risk,
                'condition_degradation_rate': np.array(traffic_impact) * 0.1,  # Impact on degradation
                'recommended_monitoring': ['Quarterly' if score > 2 else 'Semi-annual' if score > 1 else 'Annual' 
                                         for score in traffic_impact]
            }, index=X.index)
            
            return results
            
        except Exception as e:
            raise ModelTrainingError(f"Traffic impact prediction failed: {e}")