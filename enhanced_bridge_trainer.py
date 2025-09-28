#!/usr/bin/env python3
"""
Enhanced Bridge Model Trainer with NTAD Dataset Integration
Trains comprehensive bridge condition prediction models using National Bridge Inventory data
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class EnhancedBridgeModelTrainer:
    """
    Enhanced bridge condition prediction trainer using comprehensive NTAD dataset
    """
    
    def __init__(self, data_dir="data/ntad", model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_data_file = self.data_dir / "nbi_processed_data.csv"
        self.bridge_data = None
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_performance = {}
        
        # Define target variables for prediction
        self.target_variables = [
            'overall_condition_score',
            'deck_condition', 
            'superstructure_condition',
            'substructure_condition',
            'maintenance_priority_score'
        ]
        
        # Define feature groups for different model types
        self.feature_groups = {
            'structural': [
                'age', 'structure_length', 'deck_width', 'bridge_area',
                'lanes_on_structure', 'length_to_width_ratio', 'skew_factor'
            ],
            'traffic': [
                'average_daily_traffic', 'traffic_per_lane', 'traffic_per_area',
                'percent_adt_truck', 'high_traffic'
            ],
            'environmental': [
                'weather_risk', 'seismic_risk', 'scour_critical_numeric'
            ],
            'economic': [
                'bridge_improvement_cost', 'cost_per_area', 'high_cost_bridge'
            ],
            'operational': [
                'fracture_critical_numeric', 'years_since_inspection',
                'overdue_inspection', 'frequent_inspection'
            ],
            'categorical': [
                'is_interstate', 'is_urban', 'old_bridge', 'large_bridge'
            ]
        }
    
    def load_data(self):
        """Load processed NTAD bridge data"""
        print("📊 Loading processed NTAD bridge dataset...")
        
        if not self.processed_data_file.exists():
            print("❌ Processed NTAD data not found. Please run ntad_bridge_integration.py first.")
            return None
        
        self.bridge_data = pd.read_csv(self.processed_data_file)
        print(f"✅ Loaded {len(self.bridge_data):,} bridge records with {len(self.bridge_data.columns)} features")
        
        return self.bridge_data
    
    def prepare_features(self, target_var):
        """Prepare features for model training"""
        print(f"🔧 Preparing features for {target_var} prediction...")
        
        # Combine all feature groups
        all_features = []
        for group_name, features in self.feature_groups.items():
            all_features.extend(features)
        
        # Ensure all features exist in the dataset
        available_features = [f for f in all_features if f in self.bridge_data.columns]
        missing_features = [f for f in all_features if f not in self.bridge_data.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
        
        print(f"✅ Using {len(available_features)} features for training")
        
        # Prepare feature matrix
        X = self.bridge_data[available_features].copy()
        y = self.bridge_data[target_var].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Remove rows with invalid target values
        valid_mask = ~(y.isna() | (y < 0))
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📊 Final dataset: {len(X)} samples, {len(X.columns)} features")
        
        return X, y, available_features
    
    def train_models(self, target_var):
        """Train multiple models for the target variable"""
        print(f"\\n🚀 Training models for {target_var}...")
        print("="*60)
        
        # Prepare data
        X, y, features = self.prepare_features(target_var)
        
        if X is None or len(X) == 0:
            print(f"❌ No valid data for {target_var}")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(f_regression, k=min(50, len(features)))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Store preprocessing objects
        self.scalers[target_var] = scaler
        self.feature_selectors[target_var] = selector
        
        # Define models to train
        models_to_train = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.1, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # Train models
        target_models = {}
        target_performance = {}
        
        for model_name, model in models_to_train.items():
            print(f"\\n🔄 Training {model_name}...")
            
            try:
                # Train model
                if model_name in ['Ridge', 'ElasticNet', 'SVR', 'MLP']:
                    model.fit(X_train_selected, y_train)
                    y_pred = model.predict(X_test_selected)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                if model_name in ['Ridge', 'ElasticNet', 'SVR', 'MLP']:
                    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                target_models[model_name] = model
                target_performance[model_name] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'CV_R²_mean': cv_mean,
                    'CV_R²_std': cv_std
                }
                
                print(f"   ✅ RMSE: {rmse:.4f}, R²: {r2:.4f}, CV R²: {cv_mean:.4f} ± {cv_std:.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed to train {model_name}: {e}")
        
        # Store models and performance
        self.models[target_var] = target_models
        self.model_performance[target_var] = target_performance
        
        # Find best model
        if target_performance:
            best_model_name = max(target_performance.keys(), 
                                 key=lambda x: target_performance[x]['R²'])
            best_performance = target_performance[best_model_name]
            
            print(f"\\n🏆 Best model for {target_var}: {best_model_name}")
            print(f"   R²: {best_performance['R²']:.4f}")
            print(f"   RMSE: {best_performance['RMSE']:.4f}")
            print(f"   MAE: {best_performance['MAE']:.4f}")
        
        return target_models, target_performance
    
    def train_all_models(self):
        """Train models for all target variables"""
        print("🚀 ENHANCED BRIDGE MODEL TRAINING WITH NTAD DATASET")
        print("="*70)
        
        # Load data
        if self.load_data() is None:
            return
        
        print(f"\\n🎯 Training models for {len(self.target_variables)} target variables...")
        
        # Train models for each target
        for target_var in self.target_variables:
            self.train_models(target_var)
        
        # Save all models
        self.save_models()
        
        # Display comprehensive performance summary
        self.display_performance_summary()
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("\\n💾 Saving trained models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for target_var in self.models:
            # Save models for this target
            target_dir = self.model_dir / target_var
            target_dir.mkdir(exist_ok=True)
            
            for model_name, model in self.models[target_var].items():
                model_file = target_dir / f"{model_name}_{timestamp}.joblib"
                joblib.dump(model, model_file)
            
            # Save preprocessing objects
            scaler_file = target_dir / f"scaler_{timestamp}.joblib"
            selector_file = target_dir / f"feature_selector_{timestamp}.joblib"
            
            if target_var in self.scalers:
                joblib.dump(self.scalers[target_var], scaler_file)
            if target_var in self.feature_selectors:
                joblib.dump(self.feature_selectors[target_var], selector_file)
        
        # Save performance metrics
        performance_file = self.model_dir / f"model_performance_{timestamp}.joblib"
        joblib.dump(self.model_performance, performance_file)
        
        print(f"✅ Models saved to {self.model_dir}")
    
    def display_performance_summary(self):
        """Display comprehensive model performance summary"""
        print("\\n" + "="*70)
        print("📈 ENHANCED BRIDGE MODEL PERFORMANCE SUMMARY")
        print("="*70)
        
        for target_var, models_perf in self.model_performance.items():
            print(f"\\n🎯 TARGET: {target_var.upper()}")
            print("-" * 50)
            
            if not models_perf:
                print("   ❌ No models trained successfully")
                continue
            
            # Sort models by R² score
            sorted_models = sorted(models_perf.items(), 
                                 key=lambda x: x[1]['R²'], reverse=True)
            
            print(f"{'Model':<15} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'CV R²':<12}")
            print("-" * 55)
            
            for model_name, perf in sorted_models:
                r2 = perf['R²']
                rmse = perf['RMSE']
                mae = perf['MAE']
                cv_r2 = perf['CV_R²_mean']
                cv_std = perf['CV_R²_std']
                
                # Color coding based on performance
                if r2 > 0.8:
                    status = "🟢"
                elif r2 > 0.6:
                    status = "🔵"
                elif r2 > 0.4:
                    status = "🟡"
                else:
                    status = "🟠"
                
                print(f"{status} {model_name:<13} {r2:<7.4f} {rmse:<7.4f} {mae:<7.4f} {cv_r2:<7.4f}±{cv_std:<.3f}")
            
            # Best model highlight
            best_model = sorted_models[0]
            print(f"\\n   🏆 Best: {best_model[0]} (R² = {best_model[1]['R²']:.4f})")
        
        # Overall summary
        print(f"\\n" + "="*70)
        print("🎯 TRAINING SUMMARY")
        print("="*70)
        
        total_models = sum(len(models) for models in self.models.values())
        avg_performance = {}
        
        for metric in ['R²', 'RMSE', 'MAE']:
            all_scores = []
            for target_perf in self.model_performance.values():
                for model_perf in target_perf.values():
                    all_scores.append(model_perf[metric])
            
            if all_scores:
                avg_performance[metric] = np.mean(all_scores)
        
        print(f"\\n📊 OVERALL STATISTICS:")
        print(f"   Total Models Trained: {total_models}")
        print(f"   Target Variables: {len(self.target_variables)}")
        print(f"   Training Dataset Size: {len(self.bridge_data):,} bridges")
        if avg_performance:
            print(f"   Average R²: {avg_performance.get('R²', 0):.4f}")
            print(f"   Average RMSE: {avg_performance.get('RMSE', 0):.4f}")
            print(f"   Average MAE: {avg_performance.get('MAE', 0):.4f}")
        
        print(f"\\n🎯 MODEL CAPABILITIES:")
        print("   ✓ Bridge condition prediction (overall, deck, superstructure, substructure)")
        print("   ✓ Maintenance priority scoring")
        print("   ✓ Multi-algorithm ensemble approach")
        print("   ✓ Feature selection and scaling")
        print("   ✓ Cross-validation performance assessment")
        print("   ✓ Real-world NTAD dataset training")
        
        print(f"\\n💡 USAGE RECOMMENDATIONS:")
        for target_var, models_perf in self.model_performance.items():
            if models_perf:
                best_model = max(models_perf.keys(), key=lambda x: models_perf[x]['R²'])
                best_r2 = models_perf[best_model]['R²']
                
                if best_r2 > 0.8:
                    recommendation = "Excellent - Ready for production use"
                elif best_r2 > 0.6:
                    recommendation = "Good - Suitable for decision support"
                elif best_r2 > 0.4:
                    recommendation = "Fair - Use with caution"
                else:
                    recommendation = "Needs improvement - Consider more data/features"
                
                print(f"   • {target_var}: {best_model} - {recommendation}")

def main():
    """Main function to run enhanced bridge model training"""
    print("🌉 ENHANCED BRIDGE MODEL TRAINING WITH NTAD DATASET")
    print("="*70)
    
    # Initialize trainer
    trainer = EnhancedBridgeModelTrainer()
    
    # Train all models
    trainer.train_all_models()
    
    print("\\n✅ ENHANCED BRIDGE MODEL TRAINING COMPLETE!")
    print("="*70)
    print("\\n🎯 NEXT STEPS:")
    print("1. Use trained models for bridge condition prediction")
    print("2. Apply models to new bridge datasets")  
    print("3. Generate maintenance recommendations")
    print("4. Integrate with bridge management systems")
    
    return trainer

if __name__ == "__main__":
    main()