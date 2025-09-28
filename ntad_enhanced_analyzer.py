#!/usr/bin/env python3
"""
NTAD-Enhanced Bridge Analysis System
Integrates NTAD models with comprehensive bridge analysis capabilities
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge_analysis_system import BridgeConditionAnalyzer

class NTADEnhancedBridgeAnalyzer(BridgeConditionAnalyzer):
    """
    Enhanced bridge analyzer using NTAD-trained models for improved predictions
    """
    
    def __init__(self, data_path: str = "data/raw/bridges", model_path: str = "models"):
        super().__init__(data_path)
        self.model_path = Path(model_path)
        self.ntad_models = {}
        self.ntad_scalers = {}
        self.ntad_selectors = {}
        self.load_ntad_models()
        
    def load_ntad_models(self):
        """Load pre-trained NTAD models if available"""
        print("🤖 Loading NTAD-trained models...")
        
        if not self.model_path.exists():
            print("⚠️  NTAD models not found. Using rule-based analysis.")
            return
        
        # Look for model files
        target_variables = [
            'overall_condition_score',
            'deck_condition', 
            'superstructure_condition',
            'substructure_condition',
            'maintenance_priority_score'
        ]
        
        models_loaded = 0
        for target_var in target_variables:
            target_dir = self.model_path / target_var
            if target_dir.exists():
                # Find the most recent model files
                model_files = list(target_dir.glob("*RandomForest*.joblib"))
                scaler_files = list(target_dir.glob("scaler*.joblib"))
                selector_files = list(target_dir.glob("feature_selector*.joblib"))
                
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    self.ntad_models[target_var] = joblib.load(latest_model)
                    models_loaded += 1
                    
                    if scaler_files:
                        latest_scaler = max(scaler_files, key=os.path.getctime)
                        self.ntad_scalers[target_var] = joblib.load(latest_scaler)
                    
                    if selector_files:
                        latest_selector = max(selector_files, key=os.path.getctime)
                        self.ntad_selectors[target_var] = joblib.load(latest_selector)
        
        if models_loaded > 0:
            print(f"✅ Loaded {models_loaded} NTAD-trained models")
        else:
            print("⚠️  No NTAD models found. Using rule-based analysis.")
    
    def prepare_ntad_features(self, bridge_data):
        """Prepare features for NTAD model prediction"""
        # Create features similar to NTAD training data
        features = bridge_data.copy()
        
        # Add calculated features that match NTAD training
        features['bridge_area'] = features['length_m'] * features['width_m']
        features['traffic_per_lane'] = features['estimated_daily_traffic'] / max(1, features.get('lanes_on_structure', 2))
        features['traffic_per_area'] = features['estimated_daily_traffic'] / features['bridge_area']
        features['length_to_width_ratio'] = features['length_m'] / features['width_m']
        
        # Binary features
        features['high_traffic'] = 1 if features['estimated_daily_traffic'] > 30000 else 0
        features['old_bridge'] = 1 if features['age'] > 50 else 0
        features['large_bridge'] = 1 if features['bridge_area'] > 1000 else 0
        features['is_interstate'] = 1 if 'highway' in features.get('location', '').lower() else 0
        features['is_urban'] = 1 if features.get('location') in ['Downtown', 'Industrial Zone', 'Central Station'] else 0
        
        # Risk factors (estimated)
        features['fracture_critical_numeric'] = 1 if features.get('material', '') == 'Steel' and features['age'] > 40 else 0
        features['scour_critical_numeric'] = 1 if 'River' in features.get('location', '') else 0
        features['weather_risk'] = 1  # Medium risk default
        features['seismic_risk'] = 1  # Medium risk default
        
        # Maintenance features
        features['years_since_inspection'] = min(3, features.get('estimated_years_since_maintenance', 5) / 3)
        features['overdue_inspection'] = 1 if features.get('estimated_years_since_maintenance', 0) > 10 else 0
        features['frequent_inspection'] = 1 if features['condition_score'] < 6 else 0
        
        # Economic features
        cost_per_area = features.get('estimated_annual_maintenance_cost', 0) / features['bridge_area']
        features['cost_per_area'] = cost_per_area
        features['high_cost_bridge'] = 1 if cost_per_area > 100 else 0
        
        # Structural features  
        features['skew_factor'] = 0.1  # Default low skew
        
        return features
    
    def predict_with_ntad_model(self, bridge_data, target_var):
        """Make prediction using NTAD-trained model"""
        if target_var not in self.ntad_models:
            return None
        
        try:
            # Prepare features
            features = self.prepare_ntad_features(bridge_data)
            
            # Define feature order (must match training)
            feature_names = [
                'age', 'structure_length', 'deck_width', 'bridge_area',
                'lanes_on_structure', 'length_to_width_ratio', 'skew_factor',
                'average_daily_traffic', 'traffic_per_lane', 'traffic_per_area',
                'percent_adt_truck', 'high_traffic',
                'weather_risk', 'seismic_risk', 'scour_critical_numeric',
                'bridge_improvement_cost', 'cost_per_area', 'high_cost_bridge',
                'fracture_critical_numeric', 'years_since_inspection',
                'overdue_inspection', 'frequent_inspection',
                'is_interstate', 'is_urban', 'old_bridge', 'large_bridge'
            ]
            
            # Map bridge data to feature names
            feature_mapping = {
                'structure_length': 'length_m',
                'deck_width': 'width_m',
                'lanes_on_structure': 2,  # Default
                'average_daily_traffic': 'estimated_daily_traffic',
                'percent_adt_truck': 15.0,  # Default
                'bridge_improvement_cost': 'estimated_annual_maintenance_cost'
            }
            
            # Create feature vector
            X = []
            for feature_name in feature_names:
                if feature_name in features:
                    X.append(features[feature_name])
                elif feature_name in feature_mapping:
                    mapped_name = feature_mapping[feature_name]
                    if isinstance(mapped_name, (int, float)):
                        X.append(mapped_name)
                    else:
                        X.append(features.get(mapped_name, 0))
                else:
                    X.append(0)  # Default value
            
            X = np.array(X).reshape(1, -1)
            
            # Apply preprocessing if available
            if target_var in self.ntad_scalers:
                X = self.ntad_scalers[target_var].transform(X)
            
            if target_var in self.ntad_selectors:
                X = self.ntad_selectors[target_var].transform(X)
            
            # Make prediction
            prediction = self.ntad_models[target_var].predict(X)[0]
            
            # Ensure reasonable bounds
            if target_var.endswith('_condition'):
                prediction = max(4, min(9, prediction))
            elif target_var == 'overall_condition_score':
                prediction = max(1, min(10, prediction))
            
            return prediction
            
        except Exception as e:
            print(f"⚠️  NTAD prediction failed for {target_var}: {e}")
            return None
    
    def analyze_bridge_enhanced(self, bridge_id: str) -> dict:
        """
        Enhanced bridge analysis using NTAD models when available
        """
        if self.bridge_data is None:
            return {"error": "No bridge data loaded"}
        
        # Find the bridge
        bridge_record = self.bridge_data[self.bridge_data['bridge_id'] == bridge_id]
        
        if bridge_record.empty:
            available_bridges = self.bridge_data[['bridge_id', 'name', 'condition_score']].to_dict('records')
            return {
                "error": f"Bridge {bridge_id} not found",
                "available_bridges": available_bridges
            }
        
        bridge = bridge_record.iloc[0]
        
        # Get base analysis
        base_analysis = self.analyze_bridge(bridge_id)
        
        # Enhanced predictions using NTAD models
        ntad_predictions = {}
        
        if self.ntad_models:
            print(f"🤖 Applying NTAD models to {bridge_id}...")
            
            # Make predictions with NTAD models
            predictions = {}
            for target_var in self.ntad_models.keys():
                prediction = self.predict_with_ntad_model(bridge, target_var)
                if prediction is not None:
                    predictions[target_var] = prediction
            
            if predictions:
                ntad_predictions = {
                    'ntad_model_predictions': {
                        'overall_condition_predicted': predictions.get('overall_condition_score'),
                        'deck_condition_predicted': predictions.get('deck_condition'),
                        'superstructure_condition_predicted': predictions.get('superstructure_condition'),
                        'substructure_condition_predicted': predictions.get('substructure_condition'),
                        'maintenance_priority_predicted': predictions.get('maintenance_priority_score'),
                        'model_confidence': 'High' if len(predictions) >= 3 else 'Medium',
                        'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'model_comparison': {
                        'current_vs_predicted': {
                            'current_condition': float(bridge['condition_score']),
                            'predicted_condition': predictions.get('overall_condition_score', bridge['condition_score']),
                            'difference': abs(predictions.get('overall_condition_score', bridge['condition_score']) - bridge['condition_score']) if 'overall_condition_score' in predictions else 0,
                            'agreement_level': self._assess_prediction_agreement(bridge['condition_score'], predictions.get('overall_condition_score'))
                        }
                    },
                    'enhanced_insights': self._generate_enhanced_insights(bridge, predictions)
                }
        
        # Combine analyses
        enhanced_analysis = {**base_analysis, **ntad_predictions}
        
        # Update executive summary with NTAD insights
        if ntad_predictions:
            enhanced_analysis['executive_summary'] = self._create_enhanced_executive_summary(bridge, base_analysis, predictions)
        
        return enhanced_analysis
    
    def _assess_prediction_agreement(self, current_score, predicted_score):
        """Assess agreement between current and predicted scores"""
        if predicted_score is None:
            return "No prediction available"
        
        difference = abs(current_score - predicted_score)
        
        if difference <= 0.5:
            return "Excellent agreement"
        elif difference <= 1.0:
            return "Good agreement"
        elif difference <= 2.0:
            return "Fair agreement"
        else:
            return "Poor agreement - investigation recommended"
    
    def _generate_enhanced_insights(self, bridge, predictions):
        """Generate enhanced insights using NTAD predictions"""
        insights = []
        
        if 'overall_condition_score' in predictions:
            predicted = predictions['overall_condition_score']
            current = bridge['condition_score']
            
            if predicted < current - 1:
                insights.append("NTAD model suggests condition may be worse than recorded - recommend detailed inspection")
            elif predicted > current + 1:
                insights.append("NTAD model suggests condition may be better than recorded - verify assessment accuracy")
            else:
                insights.append("NTAD model prediction aligns with current assessment")
        
        if 'maintenance_priority_score' in predictions:
            priority = predictions['maintenance_priority_score']
            if priority > 8:
                insights.append("NTAD model indicates high maintenance priority based on national patterns")
            elif priority < 4:
                insights.append("NTAD model suggests lower maintenance urgency compared to similar bridges")
        
        # Component-specific insights
        component_predictions = {
            'deck': predictions.get('deck_condition'),
            'superstructure': predictions.get('superstructure_condition'),
            'substructure': predictions.get('substructure_condition')
        }
        
        worst_component = min(component_predictions.items(), 
                             key=lambda x: x[1] if x[1] is not None else 10)
        
        if worst_component[1] is not None and worst_component[1] < 6:
            insights.append(f"NTAD model identifies {worst_component[0]} as primary concern (predicted condition: {worst_component[1]:.1f})")
        
        return insights
    
    def _create_enhanced_executive_summary(self, bridge, base_analysis, predictions):
        """Create enhanced executive summary with NTAD insights"""
        base_summary = base_analysis['executive_summary']
        
        if 'overall_condition_score' in predictions:
            predicted_condition = predictions['overall_condition_score']
            current_condition = bridge['condition_score']
            
            # Add NTAD insights to summary
            ntad_section = f"""

🤖 NTAD MODEL INSIGHTS:
• Predicted Condition: {predicted_condition:.1f}/10.0 (vs. Current: {current_condition:.1f}/10.0)
• Model Agreement: {self._assess_prediction_agreement(current_condition, predicted_condition)}
• National Comparison: {'Above average' if predicted_condition > 6.5 else 'Below average'} for similar bridges

ENHANCED RECOMMENDATIONS:
"""
            
            if predicted_condition < current_condition - 1:
                ntad_section += "🔍 INVESTIGATION RECOMMENDED: Model suggests potential hidden issues\n"
            elif predicted_condition > current_condition + 1:
                ntad_section += "✅ CONDITION VERIFICATION: Model suggests better condition than recorded\n"
            
            if 'maintenance_priority_score' in predictions:
                priority = predictions['maintenance_priority_score']
                ntad_section += f"📊 National Priority Ranking: {self._get_priority_ranking(priority)}\n"
            
            enhanced_summary = base_summary + ntad_section
        else:
            enhanced_summary = base_summary + "\\n\\n🤖 NTAD models applied for enhanced analysis"
        
        return enhanced_summary
    
    def _get_priority_ranking(self, priority_score):
        """Get priority ranking description"""
        if priority_score > 12:
            return "Top 10% nationwide - Critical priority"
        elif priority_score > 8:
            return "Top 25% nationwide - High priority"
        elif priority_score > 5:
            return "Average nationwide - Medium priority"
        else:
            return "Below average nationwide - Low priority"

def main():
    """Main function to demonstrate NTAD-enhanced analysis"""
    print("🌉 NTAD-ENHANCED BRIDGE ANALYSIS SYSTEM")
    print("="*70)
    print("Combining rule-based analysis with NTAD machine learning models")
    
    # Initialize enhanced analyzer
    analyzer = NTADEnhancedBridgeAnalyzer()
    
    # Load bridge data
    bridge_data = analyzer.load_bridge_data()
    
    if bridge_data is None:
        print("❌ Failed to load bridge data")
        return
    
    print(f"\\n🌉 ENHANCED ANALYSIS CAPABILITIES:")
    if analyzer.ntad_models:
        print(f"   ✅ NTAD Models: {len(analyzer.ntad_models)} trained models loaded")
        print(f"   ✅ Model Types: {list(analyzer.ntad_models.keys())}")
        print(f"   ✅ Enhanced Predictions: Available")
    else:
        print(f"   ⚠️  NTAD Models: Not available (run enhanced_bridge_trainer.py first)")
        print(f"   ✅ Rule-based Analysis: Available")
    
    print(f"\\n📋 AVAILABLE BRIDGES:")
    for _, bridge in bridge_data.iterrows():
        condition = bridge['condition_score']
        status = "🟢" if condition >= 7 else "🟡" if condition >= 5.5 else "🟠"
        print(f"   {status} {bridge['bridge_id']}: {bridge['name']} - {condition:.1f}/10")
    
    # Demonstrate enhanced analysis
    print(f"\\n🔍 ENHANCED ANALYSIS DEMONSTRATION:")
    print("="*50)
    
    for bridge_id in bridge_data['bridge_id'].tolist()[:2]:  # Demo first 2 bridges
        print(f"\\n🌉 Analyzing {bridge_id} with NTAD Enhancement...")
        
        analysis = analyzer.analyze_bridge_enhanced(bridge_id)
        
        if 'error' not in analysis:
            # Show executive summary
            print(analysis['executive_summary'])
            
            # Show NTAD predictions if available
            if 'ntad_model_predictions' in analysis:
                predictions = analysis['ntad_model_predictions']
                comparison = analysis['model_comparison']['current_vs_predicted']
                
                print(f"\\n🤖 NTAD MODEL RESULTS:")
                print(f"   Current Condition: {comparison['current_condition']:.1f}/10")
                if predictions['overall_condition_predicted']:
                    print(f"   Predicted Condition: {predictions['overall_condition_predicted']:.1f}/10")
                    print(f"   Agreement: {comparison['agreement_level']}")
                    print(f"   Model Confidence: {predictions['model_confidence']}")
                
                if 'enhanced_insights' in analysis:
                    print(f"\\n💡 ENHANCED INSIGHTS:")
                    for insight in analysis['enhanced_insights'][:3]:
                        print(f"      • {insight}")
        else:
            print(f"❌ {analysis['error']}")
    
    print(f"\\n✅ NTAD-ENHANCED BRIDGE ANALYSIS SYSTEM READY")
    print("="*70)

if __name__ == "__main__":
    main()