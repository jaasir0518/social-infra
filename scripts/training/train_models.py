#!/usr/bin/env python3
"""
Training script for social infrastructure prediction models.

This script handles the complete model training pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training for each infrastructure type
4. Model evaluation and validation
5. Model saving and artifact management
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from config.settings import settings
from data.data_loader import DataLoader
from features.feature_engineering import FeatureEngineer
from models.bridge_prediction import BridgePredictor
from models.housing_prediction import HousingPredictor
from models.road_prediction import RoadPredictor
from models.ensemble_model import EnsembleModel
from evaluation.metrics import ModelEvaluator
from utils.helpers import ensure_directory, save_model
from utils.exceptions import ModelTrainingError


class ModelTrainer:
    """
    Handles training of all infrastructure prediction models.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = settings
        self.data_loader = DataLoader(base_path=self.config.data.raw_data_path)
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.monitoring.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.monitoring.mlflow_experiment_name)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_and_prepare_data(
        self,
        infrastructure_type: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for a specific infrastructure type.
        
        Args:
            infrastructure_type: Type of infrastructure to load data for
            
        Returns:
            Tuple of features DataFrame and target Series
        """
        try:
            self.logger.info(f"Loading data for {infrastructure_type}")
            
            # Load raw data
            raw_data = self.data_loader.load_infrastructure_data(infrastructure_type)
            
            # Feature engineering
            self.logger.info("Engineering features...")
            features_df = self.feature_engineer.create_features(
                raw_data, infrastructure_type
            )
            
            # Extract target variable
            target_col = self._get_target_column(infrastructure_type)
            if target_col not in features_df.columns:
                raise ValueError(f"Target column '{target_col}' not found")
            
            X = features_df.drop(columns=[target_col])
            y = features_df[target_col]
            
            self.logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to prepare data for {infrastructure_type}: {e}")
    
    def train_model(
        self,
        infrastructure_type: str,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Any:
        """
        Train a model for specific infrastructure type.
        
        Args:
            infrastructure_type: Type of infrastructure
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Trained model
        """
        try:
            self.logger.info(f"Training {infrastructure_type} model...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.data.test_size,
                random_state=self.config.data.random_state,
                stratify=y if self._is_classification_task(infrastructure_type) else None
            )
            
            # Initialize model
            model = self._get_model_instance(infrastructure_type)
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"{infrastructure_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params(self._get_model_params(infrastructure_type))
                mlflow.log_param("infrastructure_type", infrastructure_type)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("features", X.shape[1])
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Evaluate model
                if self._is_classification_task(infrastructure_type):
                    train_metrics = self.evaluator.evaluate_classification(y_train, y_pred_train)
                    test_metrics = self.evaluator.evaluate_classification(y_test, y_pred_test)
                else:
                    train_metrics = self.evaluator.evaluate_regression(y_train, y_pred_train)
                    test_metrics = self.evaluator.evaluate_regression(y_test, y_pred_test)
                
                # Log metrics
                for metric_name, value in train_metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", value)
                
                for metric_name, value in test_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", value)
                
                # Cross-validation
                if self.config.data.cv_folds > 1:
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=self.config.data.cv_folds,
                        scoring='accuracy' if self._is_classification_task(infrastructure_type) else 'neg_mean_squared_error'
                    )
                    mlflow.log_metric("cv_mean_score", cv_scores.mean())
                    mlflow.log_metric("cv_std_score", cv_scores.std())
                
                # Log model
                mlflow.sklearn.log_model(model, f"{infrastructure_type}_model")
                
                # Print results
                self.logger.info(f"Model training completed for {infrastructure_type}")
                self.logger.info(f"Test accuracy/R²: {test_metrics.get('accuracy', test_metrics.get('r2_score', 'N/A')):.4f}")
                
                return model
                
        except Exception as e:
            raise ModelTrainingError(f"Failed to train {infrastructure_type} model: {e}")
    
    def train_ensemble_model(
        self,
        trained_models: Dict[str, Any],
        data_dict: Dict[str, Tuple[pd.DataFrame, pd.Series]]
    ) -> EnsembleModel:
        """
        Train ensemble model combining individual models.
        
        Args:
            trained_models: Dictionary of trained individual models
            data_dict: Dictionary of data for each infrastructure type
            
        Returns:
            Trained ensemble model
        """
        try:
            self.logger.info("Training ensemble model...")
            
            with mlflow.start_run(run_name=f"ensemble_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Initialize ensemble
                ensemble = EnsembleModel(
                    models=trained_models,
                    method=self.config.models.ensemble_method,
                    weights=self.config.models.ensemble_weights
                )
                
                # Prepare ensemble data (simplified approach)
                # In practice, you might want more sophisticated ensemble training
                ensemble_X = []
                ensemble_y = []
                
                for infra_type, (X, y) in data_dict.items():
                    # Sample from each dataset
                    sample_size = min(1000, len(X))
                    indices = np.random.choice(len(X), sample_size, replace=False)
                    ensemble_X.append(X.iloc[indices])
                    ensemble_y.append(y.iloc[indices])
                
                # Combine data
                combined_X = pd.concat(ensemble_X, ignore_index=True)
                combined_y = pd.concat(ensemble_y, ignore_index=True)
                
                # Train ensemble
                ensemble.fit(combined_X, combined_y)
                
                # Evaluate ensemble
                predictions = ensemble.predict(combined_X)
                if self._is_classification_task('ensemble'):
                    metrics = self.evaluator.evaluate_classification(combined_y, predictions)
                else:
                    metrics = self.evaluator.evaluate_regression(combined_y, predictions)
                
                # Log metrics
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"ensemble_{metric_name}", value)
                
                # Log ensemble model
                mlflow.sklearn.log_model(ensemble, "ensemble_model")
                
                self.logger.info("Ensemble model training completed")
                
                return ensemble
                
        except Exception as e:
            raise ModelTrainingError(f"Failed to train ensemble model: {e}")
    
    def save_models(
        self,
        models: Dict[str, Any],
        model_dir: str = None
    ) -> None:
        """
        Save trained models to disk.
        
        Args:
            models: Dictionary of trained models
            model_dir: Directory to save models
        """
        if model_dir is None:
            model_dir = self.config.models_dir / "trained_models"
        
        model_dir = Path(model_dir)
        ensure_directory(model_dir)
        
        for model_name, model in models.items():
            model_path = model_dir / f"{model_name}_model.pkl"
            save_model(model, model_path)
            self.logger.info(f"Saved {model_name} model to {model_path}")
    
    def _get_model_instance(self, infrastructure_type: str) -> Any:
        """Get model instance for infrastructure type."""
        model_classes = {
            'bridge': BridgePredictor,
            'housing': HousingPredictor,
            'road': RoadPredictor
        }
        
        model_class = model_classes.get(infrastructure_type)
        if model_class is None:
            raise ValueError(f"Unknown infrastructure type: {infrastructure_type}")
        
        return model_class()
    
    def _get_model_params(self, infrastructure_type: str) -> Dict[str, Any]:
        """Get model parameters for infrastructure type."""
        param_mapping = {
            'bridge': self.config.models.bridge_params,
            'housing': self.config.models.housing_params,
            'road': self.config.models.road_params
        }
        
        return param_mapping.get(infrastructure_type, {})
    
    def _get_target_column(self, infrastructure_type: str) -> str:
        """Get target column name for infrastructure type."""
        target_mapping = {
            'bridge': 'condition_score',
            'housing': 'condition_rating',
            'road': 'condition_score'
        }
        
        return target_mapping.get(infrastructure_type, 'target')
    
    def _is_classification_task(self, infrastructure_type: str) -> bool:
        """Check if task is classification or regression."""
        # For simplicity, assume all tasks are regression
        # In practice, you might have different task types
        return False
    
    def run_training_pipeline(self, infrastructure_types: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            infrastructure_types: List of infrastructure types to train
            
        Returns:
            Dictionary of trained models
        """
        if infrastructure_types is None:
            infrastructure_types = ['bridge', 'housing', 'road']
        
        trained_models = {}
        data_dict = {}
        
        try:
            # Train individual models
            for infra_type in infrastructure_types:
                self.logger.info(f"Processing {infra_type} infrastructure...")
                
                # Load and prepare data
                X, y = self.load_and_prepare_data(infra_type)
                data_dict[infra_type] = (X, y)
                
                # Train model
                model = self.train_model(infra_type, X, y)
                trained_models[infra_type] = model
            
            # Train ensemble model
            if len(trained_models) > 1:
                ensemble_model = self.train_ensemble_model(trained_models, data_dict)
                trained_models['ensemble'] = ensemble_model
            
            # Save all models
            self.save_models(trained_models)
            
            self.logger.info("Training pipeline completed successfully")
            
            return trained_models
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train infrastructure prediction models')
    parser.add_argument(
        '--infrastructure-types',
        nargs='+',
        default=['bridge', 'housing', 'road'],
        help='Infrastructure types to train models for'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(config_path=args.config)
    
    # Run training pipeline
    trained_models = trainer.run_training_pipeline(args.infrastructure_types)
    
    print(f"Training completed. Trained {len(trained_models)} models:")
    for model_name in trained_models.keys():
        print(f"  - {model_name}")


if __name__ == "__main__":
    main()