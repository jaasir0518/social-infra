"""
Model evaluation metrics and utilities.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.exceptions import EvaluationError
from config.logging_config import LoggingMixin


class ModelEvaluator(LoggingMixin):
    """
    Comprehensive model evaluation utilities.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_history = []
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Optional model name for logging
            
        Returns:
            Dictionary of regression metrics
        """
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred),
                'mape': self._calculate_mape(y_true, y_pred),
                'max_error': np.max(np.abs(y_true - y_pred)),
                'std_error': np.std(y_true - y_pred)
            }
            
            # Store evaluation result
            evaluation_result = {
                'model_name': model_name or 'unknown',
                'task_type': 'regression',
                'metrics': metrics,
                'n_samples': len(y_true)
            }
            self.evaluation_history.append(evaluation_result)
            
            if model_name:
                self.logger.info(f"Regression evaluation for {model_name}: R² = {metrics['r2_score']:.4f}, RMSE = {metrics['rmse']:.4f}")
            
            return metrics
            
        except Exception as e:
            raise EvaluationError(f"Regression evaluation failed: {e}")
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: Optional[str] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Optional model name for logging
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of classification metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
                'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
            }
            
            # Store evaluation result
            evaluation_result = {
                'model_name': model_name or 'unknown',
                'task_type': 'classification',
                'metrics': metrics,
                'n_samples': len(y_true)
            }
            self.evaluation_history.append(evaluation_result)
            
            if model_name:
                self.logger.info(f"Classification evaluation for {model_name}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            raise EvaluationError(f"Classification evaluation failed: {e}")
    
    def compare_models(
        self,
        results: List[Dict[str, Any]],
        primary_metric: str = 'r2_score'
    ) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            results: List of evaluation results
            primary_metric: Primary metric for ranking
            
        Returns:
            DataFrame with model comparison
        """
        try:
            comparison_data = []
            
            for result in results:
                model_data = {
                    'model_name': result.get('model_name', 'unknown'),
                    'task_type': result.get('task_type', 'unknown'),
                    'n_samples': result.get('n_samples', 0)
                }
                model_data.update(result.get('metrics', {}))
                comparison_data.append(model_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Sort by primary metric (descending for most metrics)
            if primary_metric in comparison_df.columns:
                ascending = primary_metric in ['mse', 'rmse', 'mae', 'max_error']  # Lower is better
                comparison_df = comparison_df.sort_values(primary_metric, ascending=ascending)
            
            return comparison_df
            
        except Exception as e:
            raise EvaluationError(f"Model comparison failed: {e}")
    
    def plot_regression_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'Model',
        save_path: Optional[str] = None
    ) -> None:
        """
        Create regression evaluation plots.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Model name for plot title
            save_path: Optional path to save plot
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Regression Results: {model_name}', fontsize=16)
            
            # Predicted vs Actual
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[0, 0].set_xlabel('True Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Predicted vs Actual')
            
            # Residuals plot
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            
            # Residuals histogram
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residuals Distribution')
            
            # Error by prediction value
            abs_errors = np.abs(residuals)
            axes[1, 1].scatter(y_pred, abs_errors, alpha=0.6)
            axes[1, 1].set_xlabel('Predicted Values')
            axes[1, 1].set_ylabel('Absolute Error')
            axes[1, 1].set_title('Absolute Error vs Predicted')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Regression plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Regression plotting failed: {e}")
    
    def plot_classification_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'Model',
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create classification evaluation plots.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Model name for plot title
            class_names: Optional class names
            save_path: Optional path to save plot
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Classification Results: {model_name}', fontsize=16)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=axes[0])
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            axes[0].set_title('Confusion Matrix')
            
            # Classification report as text
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            
            # Create a simple bar plot of metrics
            metrics = ['precision', 'recall', 'f1-score']
            if 'weighted avg' in report:
                values = [report['weighted avg'][metric] for metric in metrics]
                axes[1].bar(metrics, values, alpha=0.7)
                axes[1].set_ylabel('Score')
                axes[1].set_title('Weighted Average Metrics')
                axes[1].set_ylim(0, 1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Classification plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Classification plotting failed: {e}")
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE score
        """
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get summary of all evaluations performed.
        
        Returns:
            DataFrame with evaluation summary
        """
        if not self.evaluation_history:
            return pd.DataFrame()
        
        summary_data = []
        for eval_result in self.evaluation_history:
            summary_row = {
                'model_name': eval_result['model_name'],
                'task_type': eval_result['task_type'],
                'n_samples': eval_result['n_samples']
            }
            summary_row.update(eval_result['metrics'])
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)