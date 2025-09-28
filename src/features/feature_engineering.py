"""
Feature engineering for social infrastructure prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression

from utils.exceptions import FeatureEngineeringError
from config.logging_config import LoggingMixin


class FeatureEngineer(LoggingMixin):
    """
    Handles feature engineering for infrastructure prediction models.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        
    def create_features(
        self,
        data_dict: Dict[str, pd.DataFrame],
        infrastructure_type: str
    ) -> pd.DataFrame:
        """
        Create features for a specific infrastructure type.
        
        Args:
            data_dict: Dictionary of raw data DataFrames
            infrastructure_type: Type of infrastructure
            
        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info(f"Creating features for {infrastructure_type}")
            
            if infrastructure_type == "bridge":
                return self._create_bridge_features(data_dict)
            elif infrastructure_type == "housing":
                return self._create_housing_features(data_dict)
            elif infrastructure_type == "road":
                return self._create_road_features(data_dict)
            else:
                raise ValueError(f"Unknown infrastructure type: {infrastructure_type}")
                
        except Exception as e:
            raise FeatureEngineeringError(f"Feature creation failed: {e}")
    
    def _create_bridge_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create bridge-specific features.
        
        Args:
            data_dict: Dictionary of raw bridge data
            
        Returns:
            DataFrame with bridge features
        """
        try:
            # Start with inventory data
            inventory = data_dict.get('inventory', pd.DataFrame())
            conditions = data_dict.get('conditions', pd.DataFrame())
            
            if inventory.empty:
                raise ValueError("Bridge inventory data is required")
            
            # Merge data
            if not conditions.empty:
                features_df = pd.merge(inventory, conditions, on='bridge_id', how='left')
            else:
                features_df = inventory.copy()
            
            # Age feature
            current_year = datetime.now().year
            if 'construction_year' in features_df.columns:
                features_df['age'] = current_year - features_df['construction_year']
                features_df['age_category'] = pd.cut(
                    features_df['age'],
                    bins=[0, 10, 25, 50, 100],
                    labels=['new', 'modern', 'mature', 'old']
                )
            
            # Size features
            if 'length_m' in features_df.columns and 'width_m' in features_df.columns:
                features_df['area_m2'] = features_df['length_m'] * features_df['width_m']
                features_df['aspect_ratio'] = features_df['length_m'] / features_df['width_m']
            
            # Material encoding
            if 'material' in features_df.columns:
                material_encoder = LabelEncoder()
                features_df['material_encoded'] = material_encoder.fit_transform(
                    features_df['material'].fillna('unknown')
                )
                self.encoders['bridge_material'] = material_encoder
            
            # Location features
            if 'latitude' in features_df.columns and 'longitude' in features_df.columns:
                features_df['location_cluster'] = self._create_location_clusters(
                    features_df[['latitude', 'longitude']]
                )
            
            # Condition-based features
            if 'condition_score' in features_df.columns:
                features_df['condition_category'] = pd.cut(
                    features_df['condition_score'],
                    bins=[0, 5, 7, 8.5, 10],
                    labels=['poor', 'fair', 'good', 'excellent']
                )
            
            # Drop non-numeric columns for modeling
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            features_df = features_df[numeric_columns]
            
            # Fill missing values
            features_df = features_df.fillna(features_df.mean())
            
            self.logger.info(f"Created {features_df.shape[1]} bridge features")
            
            return features_df
            
        except Exception as e:
            raise FeatureEngineeringError(f"Bridge feature creation failed: {e}")
    
    def _create_housing_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create housing-specific features.
        
        Args:
            data_dict: Dictionary of raw housing data
            
        Returns:
            DataFrame with housing features
        """
        try:
            # Start with board data
            board_data = data_dict.get('board_data', pd.DataFrame())
            property_records = data_dict.get('property_records', pd.DataFrame())
            
            if board_data.empty:
                raise ValueError("Housing board data is required")
            
            features_df = board_data.copy()
            
            # Age feature
            current_year = datetime.now().year
            if 'construction_year' in features_df.columns:
                features_df['age'] = current_year - features_df['construction_year']
            
            # Property type encoding
            if 'property_type' in features_df.columns:
                property_encoder = LabelEncoder()
                features_df['property_type_encoded'] = property_encoder.fit_transform(
                    features_df['property_type'].fillna('unknown')
                )
                self.encoders['housing_property_type'] = property_encoder
            
            # Value per unit
            if 'assessed_value' in features_df.columns and 'units' in features_df.columns:
                features_df['value_per_unit'] = features_df['assessed_value'] / (
                    features_df['units'] + 1  # Add 1 to avoid division by zero
                )
            
            # Renovation recency
            if 'last_renovation' in features_df.columns:
                features_df['years_since_renovation'] = current_year - pd.to_numeric(
                    features_df['last_renovation'], errors='coerce'
                )
            
            # Drop non-numeric columns
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            features_df = features_df[numeric_columns]
            
            # Fill missing values
            features_df = features_df.fillna(features_df.mean())
            
            self.logger.info(f"Created {features_df.shape[1]} housing features")
            
            return features_df
            
        except Exception as e:
            raise FeatureEngineeringError(f"Housing feature creation failed: {e}")
    
    def _create_road_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create road-specific features.
        
        Args:
            data_dict: Dictionary of raw road data
            
        Returns:
            DataFrame with road features
        """
        try:
            # Start with network data
            network = data_dict.get('network', pd.DataFrame())
            traffic = data_dict.get('traffic', pd.DataFrame())
            conditions = data_dict.get('conditions', pd.DataFrame())
            
            if network.empty:
                raise ValueError("Road network data is required")
            
            features_df = network.copy()
            
            # Merge traffic data if available
            if not traffic.empty and 'road_id' in traffic.columns:
                features_df = pd.merge(features_df, traffic, on='road_id', how='left')
            
            # Merge conditions data if available
            if not conditions.empty and 'road_id' in conditions.columns:
                features_df = pd.merge(features_df, conditions, on='road_id', how='left')
            
            # Create synthetic features for demo
            np.random.seed(42)
            n_roads = len(features_df)
            
            features_df['length_km'] = np.random.uniform(0.5, 15.0, n_roads)
            features_df['lanes'] = np.random.randint(2, 6, n_roads)
            features_df['traffic_volume'] = np.random.uniform(1000, 50000, n_roads)
            features_df['age'] = np.random.randint(5, 40, n_roads)
            features_df['condition_score'] = np.random.uniform(4.0, 9.0, n_roads)
            
            # Traffic density
            features_df['traffic_density'] = features_df['traffic_volume'] / features_df['length_km']
            
            # Lane utilization
            features_df['traffic_per_lane'] = features_df['traffic_volume'] / features_df['lanes']
            
            # Age categories
            features_df['age_category'] = pd.cut(
                features_df['age'],
                bins=[0, 10, 20, 35, 100],
                labels=[0, 1, 2, 3]  # Use numeric labels
            ).astype(float)
            
            # Drop non-numeric columns
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            features_df = features_df[numeric_columns]
            
            # Fill missing values
            features_df = features_df.fillna(features_df.mean())
            
            self.logger.info(f"Created {features_df.shape[1]} road features")
            
            return features_df
            
        except Exception as e:
            raise FeatureEngineeringError(f"Road feature creation failed: {e}")
    
    def _create_location_clusters(self, location_df: pd.DataFrame, n_clusters: int = 5) -> pd.Series:
        """
        Create location clusters from latitude/longitude.
        
        Args:
            location_df: DataFrame with latitude and longitude columns
            n_clusters: Number of clusters to create
            
        Returns:
            Series with cluster labels
        """
        try:
            from sklearn.cluster import KMeans
            
            # Handle missing values
            clean_locations = location_df.dropna()
            
            if len(clean_locations) < n_clusters:
                # Not enough data for clustering
                return pd.Series(0, index=location_df.index)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(clean_locations)
            
            # Create series with original index
            cluster_series = pd.Series(index=location_df.index, dtype=float)
            cluster_series.loc[clean_locations.index] = clusters
            cluster_series = cluster_series.fillna(0)  # Fill missing with cluster 0
            
            return cluster_series
            
        except ImportError:
            # Fallback if sklearn is not available
            return pd.Series(0, index=location_df.index)
        except Exception as e:
            self.logger.warning(f"Location clustering failed: {e}")
            return pd.Series(0, index=location_df.index)
    
    def scale_features(
        self,
        features_df: pd.DataFrame,
        infrastructure_type: str,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            features_df: DataFrame with features to scale
            infrastructure_type: Type of infrastructure
            fit: Whether to fit the scaler or use existing one
            
        Returns:
            DataFrame with scaled features
        """
        try:
            scaler_key = f"{infrastructure_type}_scaler"
            
            if fit or scaler_key not in self.scalers:
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features_df)
                self.scalers[scaler_key] = scaler
            else:
                scaler = self.scalers[scaler_key]
                scaled_features = scaler.transform(features_df)
            
            return pd.DataFrame(
                scaled_features,
                columns=features_df.columns,
                index=features_df.index
            )
            
        except Exception as e:
            raise FeatureEngineeringError(f"Feature scaling failed: {e}")
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        infrastructure_type: str,
        k: int = 10
    ) -> pd.DataFrame:
        """
        Select top k features using univariate selection.
        
        Args:
            X: Features DataFrame
            y: Target Series
            infrastructure_type: Type of infrastructure
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        try:
            selector_key = f"{infrastructure_type}_selector"
            
            # Ensure k doesn't exceed number of features
            k = min(k, X.shape[1])
            
            selector = SelectKBest(score_func=f_regression, k=k)
            selected_features = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_columns = X.columns[selector.get_support()]
            
            # Store selector
            self.feature_selectors[selector_key] = selector
            
            return pd.DataFrame(
                selected_features,
                columns=selected_columns,
                index=X.index
            )
            
        except Exception as e:
            raise FeatureEngineeringError(f"Feature selection failed: {e}")