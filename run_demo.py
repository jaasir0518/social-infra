#!/usr/bin/env python3
"""
Simple demo script to run infrastructure prediction models.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import settings
from models.bridge_prediction import BridgePredictor
from models.housing_prediction import HousingPredictor
from evaluation.metrics import ModelEvaluator

def create_sample_data(n_samples=1000):
    """Create sample infrastructure data for demonstration."""
    np.random.seed(42)
    
    # Bridge data
    bridge_data = pd.DataFrame({
        'age': np.random.randint(1, 100, n_samples),
        'traffic_volume': np.random.randint(100, 10000, n_samples),
        'material_type': np.random.choice(['concrete', 'steel', 'wood'], n_samples),
        'length': np.random.uniform(10, 1000, n_samples),
        'width': np.random.uniform(5, 50, n_samples),
        'last_maintenance': np.random.randint(0, 10, n_samples),
        'latitude': np.random.uniform(30, 50, n_samples),
        'longitude': np.random.uniform(-120, -70, n_samples)
    })
    
    # Create target: condition score (0-10, higher is better)
    bridge_data['condition_score'] = (
        10 - 0.05 * bridge_data['age'] 
        - 0.0001 * bridge_data['traffic_volume']
        + 0.5 * bridge_data['last_maintenance']
        + np.random.normal(0, 1, n_samples)
    ).clip(0, 10)
    
    # Housing data
    housing_data = pd.DataFrame({
        'age': np.random.randint(1, 150, n_samples),
        'size_sqft': np.random.randint(500, 5000, n_samples),
        'num_units': np.random.randint(1, 200, n_samples),
        'property_type': np.random.choice(['residential', 'commercial', 'mixed'], n_samples),
        'last_renovation': np.random.randint(0, 30, n_samples),
        'population_density': np.random.uniform(100, 5000, n_samples),
        'latitude': np.random.uniform(30, 50, n_samples),
        'longitude': np.random.uniform(-120, -70, n_samples)
    })
    
    # Create target: condition rating (0-5, higher is better)
    housing_data['condition_rating'] = (
        5 - 0.02 * housing_data['age']
        + 0.1 * housing_data['last_renovation']
        - 0.0001 * housing_data['num_units']
        + np.random.normal(0, 0.5, n_samples)
    ).clip(0, 5)
    
    return bridge_data, housing_data

def run_bridge_model_demo():
    """Run bridge prediction model demo."""
    print("=" * 60)
    print("🌉 BRIDGE INFRASTRUCTURE PREDICTION DEMO")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample bridge data...")
    bridge_data, _ = create_sample_data(1000)
    print(f"   Generated {len(bridge_data)} bridge records")
    
    # Prepare features and target
    feature_cols = ['age', 'traffic_volume', 'length', 'width', 'last_maintenance', 'latitude', 'longitude']
    X = bridge_data[feature_cols]
    y = bridge_data['condition_score']
    
    # Handle categorical data (simple encoding for demo)
    material_encoding = {'concrete': 0, 'steel': 1, 'wood': 2}
    X['material_encoded'] = bridge_data['material_type'].map(material_encoding)
    X = X.drop(columns=['material_type']) if 'material_type' in X.columns else X
    
    print("📈 Features shape:", X.shape)
    print("🎯 Target range:", f"{y.min():.2f} - {y.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"🔄 Split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train model
    print("🚀 Training bridge prediction model...")
    bridge_model = BridgePredictor()
    bridge_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = bridge_model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("📊 Model Performance:")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   Mean target: {y_test.mean():.3f}")
    print(f"   Relative error: {(rmse/y_test.mean()*100):.1f}%")
    
    # Show some predictions
    print("🔍 Sample Predictions:")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        print(f"   Bridge {i+1}: Actual={actual:.2f}, Predicted={predicted:.2f}, Diff={abs(actual-predicted):.2f}")
    
    return bridge_model, rmse

def run_housing_model_demo():
    """Run housing prediction model demo."""
    print("\\n" + "=" * 60)
    print("🏠 HOUSING INFRASTRUCTURE PREDICTION DEMO")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample housing data...")
    _, housing_data = create_sample_data(1000)
    print(f"   Generated {len(housing_data)} housing records")
    
    # Prepare features and target
    feature_cols = ['age', 'size_sqft', 'num_units', 'last_renovation', 'population_density', 'latitude', 'longitude']
    X = housing_data[feature_cols]
    y = housing_data['condition_rating']
    
    # Handle categorical data
    property_encoding = {'residential': 0, 'commercial': 1, 'mixed': 2}
    X['property_encoded'] = housing_data['property_type'].map(property_encoding)
    
    print("📈 Features shape:", X.shape)
    print("🎯 Target range:", f"{y.min():.2f} - {y.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"🔄 Split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train model
    print("🚀 Training housing prediction model...")
    housing_model = HousingPredictor()
    housing_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = housing_model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("📊 Model Performance:")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   Mean target: {y_test.mean():.3f}")
    print(f"   Relative error: {(rmse/y_test.mean()*100):.1f}%")
    
    # Show some predictions
    print("🔍 Sample Predictions:")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        print(f"   Property {i+1}: Actual={actual:.2f}, Predicted={predicted:.2f}, Diff={abs(actual-predicted):.2f}")
    
    return housing_model, rmse

def main():
    """Main demo function."""
    print("🏗️  Social Infrastructure Prediction System")
    print("   Demonstrating ML models for infrastructure maintenance planning")
    print()
    
    try:
        # Run bridge model demo
        bridge_model, bridge_rmse = run_bridge_model_demo()
        
        # Run housing model demo  
        housing_model, housing_rmse = run_housing_model_demo()
        
        # Summary
        print("\\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(f"✅ Bridge Model RMSE: {bridge_rmse:.3f}")
        print(f"✅ Housing Model RMSE: {housing_rmse:.3f}")
        print("\\n🎉 Demo completed successfully!")
        print("\\nNext steps:")
        print("   • Use real data from data/raw/ directory")
        print("   • Run full training: python scripts/training/train_models.py")
        print("   • Start API server: python api/app.py")
        print("   • Explore notebooks in notebooks/ directory")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()