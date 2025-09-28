#!/usr/bin/env python3
"""
Advanced Infrastructure Analysis and Training System
Provides comprehensive historical analysis and condition assessment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import settings
from models.bridge_prediction import BridgePredictor
from models.housing_prediction import HousingPredictor
from models.road_prediction import RoadPredictor

class InfrastructureAnalyzer:
    """
    Advanced infrastructure analyzer that provides comprehensive
    historical analysis and condition assessment
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.historical_data = {}
        self.condition_thresholds = {
            'bridge': {
                'excellent': 8.5, 'good': 7.0, 'fair': 5.5, 
                'poor': 4.0, 'critical': 2.5
            },
            'housing': {
                'excellent': 4.5, 'good': 3.5, 'fair': 2.5,
                'poor': 1.5, 'critical': 0.8
            },
            'road': {
                'excellent': 8.0, 'good': 6.5, 'fair': 5.0,
                'poor': 3.5, 'critical': 2.0
            }
        }
    
    def generate_comprehensive_dataset(self, n_samples=5000):
        """Generate comprehensive infrastructure dataset with historical data"""
        np.random.seed(42)
        
        # Generate time series data (5 years of monthly data)
        dates = pd.date_range(start='2019-01-01', end='2024-12-31', freq='M')
        
        datasets = {}
        
        # Bridge Dataset
        print("📊 Generating comprehensive bridge dataset...")
        bridge_data = []
        
        for i in range(n_samples):
            bridge_id = f"BR_{i:05d}"
            construction_year = np.random.randint(1950, 2020)
            current_age = 2024 - construction_year
            
            # Location factors
            lat = np.random.uniform(25, 49)  # US latitude range
            lon = np.random.uniform(-125, -66)  # US longitude range
            climate_zone = self._get_climate_zone(lat, lon)
            
            # Bridge characteristics
            bridge_type = np.random.choice(['beam', 'arch', 'suspension', 'cable', 'truss'])
            material = np.random.choice(['concrete', 'steel', 'composite', 'wood'])
            length = np.random.lognormal(4, 1)  # Log-normal for realistic distribution
            width = np.random.uniform(8, 50)
            lanes = np.random.choice([2, 4, 6, 8], p=[0.4, 0.35, 0.2, 0.05])
            
            # Traffic and usage
            daily_traffic = np.random.lognormal(8, 1.5)
            heavy_vehicle_pct = np.random.uniform(5, 25)
            
            # Generate historical condition data
            base_condition = 9.0  # New bridge starts at 9.0
            deterioration_rate = self._calculate_deterioration_rate(
                material, climate_zone, daily_traffic, heavy_vehicle_pct
            )
            
            # Maintenance history
            maintenance_events = self._generate_maintenance_history(
                construction_year, current_age, deterioration_rate
            )
            
            # Current condition calculation
            current_condition = self._calculate_current_condition(
                base_condition, current_age, deterioration_rate, maintenance_events
            )
            
            bridge_data.append({
                'id': bridge_id,
                'type': 'bridge',
                'construction_year': construction_year,
                'age': current_age,
                'latitude': lat,
                'longitude': lon,
                'climate_zone': climate_zone,
                'bridge_type': bridge_type,
                'material': material,
                'length_m': length,
                'width_m': width,
                'lanes': lanes,
                'daily_traffic': daily_traffic,
                'heavy_vehicle_pct': heavy_vehicle_pct,
                'maintenance_count': len(maintenance_events),
                'last_maintenance_years': min([2024 - m['year'] for m in maintenance_events] + [999]),
                'total_maintenance_cost': sum([m['cost'] for m in maintenance_events]),
                'condition_score': current_condition,
                'maintenance_history': maintenance_events
            })
        
        datasets['bridge'] = pd.DataFrame(bridge_data)
        
        # Housing Dataset
        print("🏠 Generating comprehensive housing dataset...")
        housing_data = []
        
        for i in range(n_samples):
            property_id = f"HU_{i:05d}"
            construction_year = np.random.randint(1900, 2020)
            current_age = 2024 - construction_year
            
            # Location factors
            lat = np.random.uniform(25, 49)
            lon = np.random.uniform(-125, -66)
            climate_zone = self._get_climate_zone(lat, lon)
            
            # Property characteristics
            property_type = np.random.choice(['single_family', 'apartment', 'condo', 'townhouse'])
            building_type = np.random.choice(['wood_frame', 'brick', 'concrete', 'steel'])
            size_sqft = np.random.lognormal(7.5, 0.6)  # Log-normal for realistic sizes
            num_floors = np.random.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.2, 0.05])
            
            if property_type == 'apartment':
                num_units = np.random.randint(4, 200)
            else:
                num_units = 1
            
            # Usage factors
            occupancy_rate = np.random.uniform(0.7, 1.0)
            population_density = np.random.lognormal(6, 1)
            
            # Generate historical condition data
            base_condition = 4.5  # New housing starts at 4.5/5
            deterioration_rate = self._calculate_housing_deterioration_rate(
                building_type, climate_zone, current_age
            )
            
            # Renovation history
            renovation_events = self._generate_renovation_history(
                construction_year, current_age
            )
            
            # Current condition calculation
            current_condition = self._calculate_housing_current_condition(
                base_condition, current_age, deterioration_rate, renovation_events
            )
            
            housing_data.append({
                'id': property_id,
                'type': 'housing',
                'construction_year': construction_year,
                'age': current_age,
                'latitude': lat,
                'longitude': lon,
                'climate_zone': climate_zone,
                'property_type': property_type,
                'building_type': building_type,
                'size_sqft': size_sqft,
                'num_floors': num_floors,
                'num_units': num_units,
                'occupancy_rate': occupancy_rate,
                'population_density': population_density,
                'renovation_count': len(renovation_events),
                'last_renovation_years': min([2024 - r['year'] for r in renovation_events] + [999]),
                'total_renovation_cost': sum([r['cost'] for r in renovation_events]),
                'condition_rating': current_condition,
                'renovation_history': renovation_events
            })
        
        datasets['housing'] = pd.DataFrame(housing_data)
        
        # Road Dataset
        print("🛣️ Generating comprehensive road dataset...")
        road_data = []
        
        for i in range(n_samples):
            road_id = f"RD_{i:05d}"
            construction_year = np.random.randint(1960, 2020)
            current_age = 2024 - construction_year
            
            # Location factors
            lat = np.random.uniform(25, 49)
            lon = np.random.uniform(-125, -66)
            climate_zone = self._get_climate_zone(lat, lon)
            
            # Road characteristics
            road_type = np.random.choice(['highway', 'arterial', 'collector', 'local'])
            surface_type = np.random.choice(['asphalt', 'concrete', 'composite'])
            length_km = np.random.lognormal(2, 1)
            lanes = np.random.choice([2, 4, 6, 8], p=[0.3, 0.4, 0.25, 0.05])
            
            # Traffic factors
            daily_traffic = np.random.lognormal(9, 1.2)
            heavy_vehicle_pct = np.random.uniform(3, 30)
            
            # Generate historical condition data
            base_condition = 8.5  # New road starts at 8.5/10
            deterioration_rate = self._calculate_road_deterioration_rate(
                surface_type, climate_zone, daily_traffic
            )
            
            # Maintenance history
            maintenance_events = self._generate_road_maintenance_history(
                construction_year, current_age, deterioration_rate
            )
            
            # Current condition calculation
            current_condition = self._calculate_road_current_condition(
                base_condition, current_age, deterioration_rate, maintenance_events
            )
            
            road_data.append({
                'id': road_id,
                'type': 'road',
                'construction_year': construction_year,
                'age': current_age,
                'latitude': lat,
                'longitude': lon,
                'climate_zone': climate_zone,
                'road_type': road_type,
                'surface_type': surface_type,
                'length_km': length_km,
                'lanes': lanes,
                'daily_traffic': daily_traffic,
                'heavy_vehicle_pct': heavy_vehicle_pct,
                'maintenance_count': len(maintenance_events),
                'last_maintenance_years': min([2024 - m['year'] for m in maintenance_events] + [999]),
                'total_maintenance_cost': sum([m['cost'] for m in maintenance_events]),
                'condition_score': current_condition,
                'maintenance_history': maintenance_events
            })
        
        datasets['road'] = pd.DataFrame(road_data)
        
        return datasets
    
    def _get_climate_zone(self, lat, lon):
        """Determine climate zone based on location"""
        if lat > 45:
            return 'cold'
        elif lat > 35:
            return 'temperate'
        else:
            return 'hot'
    
    def _calculate_deterioration_rate(self, material, climate_zone, traffic, heavy_vehicle_pct):
        """Calculate bridge deterioration rate"""
        base_rate = 0.1  # Base deterioration per year
        
        # Material factors
        material_factors = {'concrete': 1.0, 'steel': 1.2, 'composite': 0.8, 'wood': 1.5}
        rate = base_rate * material_factors.get(material, 1.0)
        
        # Climate factors
        climate_factors = {'cold': 1.3, 'temperate': 1.0, 'hot': 1.1}
        rate *= climate_factors.get(climate_zone, 1.0)
        
        # Traffic factors
        rate *= (1 + traffic / 50000)  # Higher traffic increases deterioration
        rate *= (1 + heavy_vehicle_pct / 100)  # Heavy vehicles increase deterioration
        
        return rate
    
    def _calculate_housing_deterioration_rate(self, building_type, climate_zone, age):
        """Calculate housing deterioration rate"""
        base_rate = 0.08  # Base deterioration per year
        
        # Building type factors
        building_factors = {'wood_frame': 1.2, 'brick': 0.8, 'concrete': 0.7, 'steel': 0.9}
        rate = base_rate * building_factors.get(building_type, 1.0)
        
        # Climate factors
        climate_factors = {'cold': 1.2, 'temperate': 1.0, 'hot': 1.1}
        rate *= climate_factors.get(climate_zone, 1.0)
        
        # Age factors (older buildings deteriorate faster)
        if age > 50:
            rate *= 1.3
        elif age > 30:
            rate *= 1.1
        
        return rate
    
    def _calculate_road_deterioration_rate(self, surface_type, climate_zone, traffic):
        """Calculate road deterioration rate"""
        base_rate = 0.15  # Base deterioration per year
        
        # Surface factors
        surface_factors = {'asphalt': 1.0, 'concrete': 0.7, 'composite': 0.8}
        rate = base_rate * surface_factors.get(surface_type, 1.0)
        
        # Climate factors
        climate_factors = {'cold': 1.4, 'temperate': 1.0, 'hot': 1.2}
        rate *= climate_factors.get(climate_zone, 1.0)
        
        # Traffic factors
        rate *= (1 + traffic / 30000)
        
        return rate
    
    def _generate_maintenance_history(self, construction_year, age, deterioration_rate):
        """Generate realistic maintenance history"""
        events = []
        current_year = construction_year
        
        # Major maintenance every 10-20 years
        while current_year + 15 < 2024:
            if np.random.random() < 0.7:  # 70% chance of maintenance
                maintenance_year = current_year + np.random.randint(10, 20)
                if maintenance_year <= 2024:
                    events.append({
                        'year': maintenance_year,
                        'type': np.random.choice(['routine', 'major', 'rehabilitation']),
                        'cost': np.random.lognormal(10, 1),  # Realistic cost distribution
                        'condition_improvement': np.random.uniform(1.0, 3.0)
                    })
                    current_year = maintenance_year
            else:
                current_year += 15
        
        return events
    
    def _generate_renovation_history(self, construction_year, age):
        """Generate realistic renovation history for housing"""
        events = []
        current_year = construction_year
        
        # Renovations every 15-25 years
        while current_year + 20 < 2024:
            if np.random.random() < 0.6:  # 60% chance of renovation
                renovation_year = current_year + np.random.randint(15, 25)
                if renovation_year <= 2024:
                    events.append({
                        'year': renovation_year,
                        'type': np.random.choice(['minor', 'major', 'complete']),
                        'cost': np.random.lognormal(9, 1),
                        'condition_improvement': np.random.uniform(0.5, 2.0)
                    })
                    current_year = renovation_year
            else:
                current_year += 20
        
        return events
    
    def _generate_road_maintenance_history(self, construction_year, age, deterioration_rate):
        """Generate realistic road maintenance history"""
        events = []
        current_year = construction_year
        
        # Road maintenance every 5-15 years
        while current_year + 8 < 2024:
            if np.random.random() < 0.8:  # 80% chance of maintenance
                maintenance_year = current_year + np.random.randint(5, 15)
                if maintenance_year <= 2024:
                    events.append({
                        'year': maintenance_year,
                        'type': np.random.choice(['resurfacing', 'patching', 'reconstruction']),
                        'cost': np.random.lognormal(11, 1),
                        'condition_improvement': np.random.uniform(2.0, 4.0)
                    })
                    current_year = maintenance_year
            else:
                current_year += 8
        
        return events
    
    def _calculate_current_condition(self, base_condition, age, deterioration_rate, maintenance_events):
        """Calculate current bridge condition"""
        condition = base_condition
        
        # Age-based deterioration
        condition -= age * deterioration_rate
        
        # Apply maintenance improvements
        for event in maintenance_events:
            years_since = 2024 - event['year']
            # Maintenance effect diminishes over time
            improvement = event['condition_improvement'] * np.exp(-years_since * 0.1)
            condition += improvement
        
        # Add some noise
        condition += np.random.normal(0, 0.3)
        
        return np.clip(condition, 0, 10)
    
    def _calculate_housing_current_condition(self, base_condition, age, deterioration_rate, renovation_events):
        """Calculate current housing condition"""
        condition = base_condition
        
        # Age-based deterioration
        condition -= age * deterioration_rate
        
        # Apply renovation improvements
        for event in renovation_events:
            years_since = 2024 - event['year']
            improvement = event['condition_improvement'] * np.exp(-years_since * 0.05)
            condition += improvement
        
        # Add some noise
        condition += np.random.normal(0, 0.2)
        
        return np.clip(condition, 0, 5)
    
    def _calculate_road_current_condition(self, base_condition, age, deterioration_rate, maintenance_events):
        """Calculate current road condition"""
        condition = base_condition
        
        # Age-based deterioration
        condition -= age * deterioration_rate
        
        # Apply maintenance improvements
        for event in maintenance_events:
            years_since = 2024 - event['year']
            improvement = event['condition_improvement'] * np.exp(-years_since * 0.15)
            condition += improvement
        
        # Add some noise
        condition += np.random.normal(0, 0.4)
        
        return np.clip(condition, 0, 10)
    
    def train_comprehensive_models(self, datasets):
        """Train comprehensive models with all features"""
        print("🚀 Training comprehensive infrastructure models...")
        
        for infra_type, data in datasets.items():
            print(f"\\n📈 Training {infra_type} model...")
            
            # Prepare features
            if infra_type == 'bridge':
                feature_cols = ['age', 'length_m', 'width_m', 'lanes', 'daily_traffic', 
                              'heavy_vehicle_pct', 'maintenance_count', 'last_maintenance_years',
                              'total_maintenance_cost', 'latitude', 'longitude']
                target_col = 'condition_score'
                categorical_cols = ['bridge_type', 'material', 'climate_zone']
            
            elif infra_type == 'housing':
                feature_cols = ['age', 'size_sqft', 'num_floors', 'num_units', 'occupancy_rate',
                              'population_density', 'renovation_count', 'last_renovation_years',
                              'total_renovation_cost', 'latitude', 'longitude']
                target_col = 'condition_rating'
                categorical_cols = ['property_type', 'building_type', 'climate_zone']
            
            else:  # road
                feature_cols = ['age', 'length_km', 'lanes', 'daily_traffic', 'heavy_vehicle_pct',
                              'maintenance_count', 'last_maintenance_years', 'total_maintenance_cost',
                              'latitude', 'longitude']
                target_col = 'condition_score'
                categorical_cols = ['road_type', 'surface_type', 'climate_zone']
            
            # Encode categorical variables
            X = data[feature_cols].copy()
            for col in categorical_cols:
                if col in data.columns:
                    le = LabelEncoder()
                    X[f'{col}_encoded'] = le.fit_transform(data[col])
                    if infra_type not in self.encoders:
                        self.encoders[infra_type] = {}
                    self.encoders[infra_type][col] = le
            
            y = data[target_col]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[infra_type] = scaler
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train ensemble model
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            print(f"   RMSE: {rmse:.3f}")
            print(f"   R² Score: {r2:.3f}")
            
            # Store model and feature columns
            self.models[infra_type] = {
                'model': model,
                'feature_cols': feature_cols,
                'categorical_cols': categorical_cols,
                'target_col': target_col
            }
            
            # Store historical data for analysis
            self.historical_data[infra_type] = data
        
        print("\\n✅ All models trained successfully!")
    
    def analyze_infrastructure(self, infra_id: str, infra_type: str) -> Dict[str, Any]:
        """
        Comprehensive infrastructure analysis with history and predictions
        """
        if infra_type not in self.historical_data:
            return {"error": f"No data available for infrastructure type: {infra_type}"}
        
        data = self.historical_data[infra_type]
        
        # Find the specific infrastructure
        record = data[data['id'] == infra_id]
        if record.empty:
            # If specific ID not found, create analysis for similar infrastructure
            print(f"⚠️ ID {infra_id} not found. Analyzing similar infrastructure...")
            record = data.sample(1)  # Get a random sample
        
        record = record.iloc[0]
        
        analysis = {
            'infrastructure_id': infra_id,
            'infrastructure_type': infra_type,
            'basic_info': self._extract_basic_info(record),
            'current_condition': self._analyze_current_condition(record, infra_type),
            'historical_analysis': self._analyze_history(record),
            'risk_assessment': self._assess_risks(record, infra_type),
            'maintenance_recommendations': self._generate_recommendations(record, infra_type),
            'future_projections': self._project_future_condition(record, infra_type),
            'comparative_analysis': self._compare_with_peers(record, data, infra_type),
            'cost_analysis': self._analyze_costs(record),
            'summary': self._generate_summary(record, infra_type)
        }
        
        return analysis
    
    def _extract_basic_info(self, record) -> Dict[str, Any]:
        """Extract basic infrastructure information"""
        return {
            'construction_year': int(record['construction_year']),
            'current_age': int(record['age']),
            'location': {
                'latitude': float(record['latitude']),
                'longitude': float(record['longitude']),
                'climate_zone': record['climate_zone']
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _analyze_current_condition(self, record, infra_type) -> Dict[str, Any]:
        """Analyze current infrastructure condition"""
        if infra_type == 'bridge':
            score = record['condition_score']
            scale = "0-10 scale"
        elif infra_type == 'housing':
            score = record['condition_rating']
            scale = "0-5 scale"
        else:  # road
            score = record['condition_score']
            scale = "0-10 scale"
        
        # Determine condition category
        thresholds = self.condition_thresholds[infra_type]
        if score >= thresholds['excellent']:
            category = 'Excellent'
            description = 'Infrastructure is in excellent condition with minimal maintenance needs'
        elif score >= thresholds['good']:
            category = 'Good'
            description = 'Infrastructure is in good condition with routine maintenance sufficient'
        elif score >= thresholds['fair']:
            category = 'Fair'
            description = 'Infrastructure shows signs of wear and may need increased maintenance'
        elif score >= thresholds['poor']:
            category = 'Poor'
            description = 'Infrastructure requires attention and significant maintenance'
        else:
            category = 'Critical'
            description = 'Infrastructure is in critical condition requiring immediate attention'
        
        return {
            'score': float(score),
            'scale': scale,
            'category': category,
            'description': description,
            'assessment_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _analyze_history(self, record) -> Dict[str, Any]:
        """Analyze historical data and trends"""
        if record['type'] == 'bridge':
            history_key = 'maintenance_history'
            event_type = 'maintenance'
        elif record['type'] == 'housing':
            history_key = 'renovation_history'
            event_type = 'renovation'
        else:  # road
            history_key = 'maintenance_history'
            event_type = 'maintenance'
        
        history = record[history_key]
        
        if not history:
            return {
                'total_events': 0,
                'recent_activity': 'No recent activity recorded',
                'trend': 'No historical data available'
            }
        
        # Analyze historical events
        events_by_decade = {}
        total_cost = 0
        recent_events = []
        
        for event in history:
            decade = (event['year'] // 10) * 10
            events_by_decade[decade] = events_by_decade.get(decade, 0) + 1
            total_cost += event['cost']
            
            if event['year'] >= 2020:
                recent_events.append(event)
        
        # Calculate trends
        if len(history) > 1:
            recent_frequency = len([e for e in history if e['year'] >= 2015]) / 10
            older_frequency = len([e for e in history if e['year'] < 2015]) / max((2015 - record['construction_year']), 1)
            
            if recent_frequency > older_frequency * 1.5:
                trend = f"Increasing {event_type} frequency - may indicate accelerated deterioration"
            elif recent_frequency < older_frequency * 0.5:
                trend = f"Decreasing {event_type} frequency - good condition or deferred maintenance"
            else:
                trend = f"Stable {event_type} pattern"
        else:
            trend = "Insufficient data for trend analysis"
        
        return {
            'total_events': len(history),
            'events_by_decade': events_by_decade,
            'total_historical_cost': float(total_cost),
            'recent_events': len(recent_events),
            'trend': trend,
            'last_event_year': max([e['year'] for e in history]) if history else None
        }
    
    def _assess_risks(self, record, infra_type) -> Dict[str, Any]:
        """Assess various risk factors"""
        risks = []
        risk_score = 0
        
        # Age-based risks
        age = record['age']
        if age > 50:
            risks.append("High age-related deterioration risk")
            risk_score += 3
        elif age > 30:
            risks.append("Moderate age-related deterioration risk")
            risk_score += 2
        
        # Climate risks
        if record['climate_zone'] == 'cold':
            risks.append("Freeze-thaw cycle damage risk")
            risk_score += 2
        elif record['climate_zone'] == 'hot':
            risks.append("Heat-related expansion/contraction stress")
            risk_score += 1
        
        # Maintenance timing risks
        last_maintenance_years = record.get('last_maintenance_years', 999)
        if last_maintenance_years > 10:
            risks.append("Overdue for maintenance")
            risk_score += 3
        elif last_maintenance_years > 5:
            risks.append("Approaching maintenance interval")
            risk_score += 1
        
        # Infrastructure-specific risks
        if infra_type == 'bridge':
            if record['daily_traffic'] > 50000:
                risks.append("High traffic load stress")
                risk_score += 2
            if record['heavy_vehicle_pct'] > 20:
                risks.append("Heavy vehicle damage risk")
                risk_score += 2
        
        elif infra_type == 'housing':
            if record['building_type'] == 'wood_frame' and record['age'] > 40:
                risks.append("Wood deterioration and pest damage risk")
                risk_score += 2
            if record['occupancy_rate'] > 0.95:
                risks.append("High usage wear and tear")
                risk_score += 1
        
        else:  # road
            if record['daily_traffic'] > 30000:
                risks.append("Heavy traffic wear")
                risk_score += 2
            if record['surface_type'] == 'asphalt' and record['age'] > 15:
                risks.append("Asphalt aging and cracking risk")
                risk_score += 2
        
        # Overall risk level
        if risk_score >= 8:
            risk_level = "High"
        elif risk_score >= 5:
            risk_level = "Moderate"
        elif risk_score >= 2:
            risk_level = "Low"
        else:
            risk_level = "Minimal"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'identified_risks': risks,
            'assessment_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _generate_recommendations(self, record, infra_type) -> List[Dict[str, Any]]:
        """Generate maintenance and improvement recommendations"""
        recommendations = []
        
        # Current condition-based recommendations
        if infra_type == 'bridge':
            score = record['condition_score']
            if score < 4:
                recommendations.append({
                    'priority': 'Critical',
                    'action': 'Immediate structural assessment and emergency repairs',
                    'timeline': 'Within 1 month',
                    'estimated_cost': 'High ($500K - $2M)'
                })
            elif score < 6:
                recommendations.append({
                    'priority': 'High',
                    'action': 'Comprehensive inspection and major maintenance',
                    'timeline': 'Within 6 months',
                    'estimated_cost': 'Moderate ($100K - $500K)'
                })
        
        # Age-based recommendations
        age = record['age']
        last_maintenance = record.get('last_maintenance_years', 999)
        
        if last_maintenance > 10:
            recommendations.append({
                'priority': 'High',
                'action': f'Overdue {infra_type} maintenance',
                'timeline': 'Within 3 months',
                'estimated_cost': 'TBD - requires inspection'
            })
        elif last_maintenance > 5:
            recommendations.append({
                'priority': 'Medium',
                'action': f'Schedule routine {infra_type} maintenance',
                'timeline': 'Within 12 months',
                'estimated_cost': 'Low-Moderate'
            })
        
        # Infrastructure-specific recommendations
        if infra_type == 'bridge' and record['material'] == 'steel':
            recommendations.append({
                'priority': 'Medium',
                'action': 'Anti-corrosion treatment and painting',
                'timeline': 'Next maintenance cycle',
                'estimated_cost': 'Moderate'
            })
        
        if not recommendations:
            recommendations.append({
                'priority': 'Low',
                'action': 'Continue routine monitoring and maintenance schedule',
                'timeline': 'As scheduled',
                'estimated_cost': 'Budget as planned'
            })
        
        return recommendations
    
    def _project_future_condition(self, record, infra_type) -> Dict[str, Any]:
        """Project future condition based on current trends"""
        current_condition = record['condition_score'] if infra_type != 'housing' else record['condition_rating']
        
        # Estimate deterioration rate based on infrastructure type and conditions
        if infra_type == 'bridge':
            base_deterioration = 0.1
            if record['daily_traffic'] > 30000:
                base_deterioration += 0.05
            if record['climate_zone'] == 'cold':
                base_deterioration += 0.03
        elif infra_type == 'housing':
            base_deterioration = 0.08
            if record['age'] > 50:
                base_deterioration += 0.04
        else:  # road
            base_deterioration = 0.15
            if record['daily_traffic'] > 20000:
                base_deterioration += 0.08
        
        # Project conditions for next 5, 10, and 20 years
        projections = {}
        for years in [5, 10, 20]:
            projected_condition = current_condition - (base_deterioration * years)
            
            # Factor in potential maintenance (assume maintenance every 10 years)
            maintenance_cycles = years // 10
            if maintenance_cycles > 0:
                projected_condition += maintenance_cycles * 2.0  # Maintenance improves condition
            
            projected_condition = max(0, projected_condition)
            
            # Determine projected category
            thresholds = self.condition_thresholds[infra_type]
            if projected_condition >= thresholds['good']:
                category = 'Good'
            elif projected_condition >= thresholds['fair']:
                category = 'Fair'
            elif projected_condition >= thresholds['poor']:
                category = 'Poor'
            else:
                category = 'Critical'
            
            projections[f'{years}_years'] = {
                'projected_score': float(projected_condition),
                'projected_category': category
            }
        
        return {
            'current_score': float(current_condition),
            'deterioration_rate_per_year': float(base_deterioration),
            'projections': projections,
            'notes': 'Projections assume regular maintenance cycles and current usage patterns'
        }
    
    def _compare_with_peers(self, record, data, infra_type) -> Dict[str, Any]:
        """Compare with similar infrastructure"""
        # Define similarity criteria
        if infra_type == 'bridge':
            similar_data = data[
                (abs(data['age'] - record['age']) <= 10) &
                (data['climate_zone'] == record['climate_zone']) &
                (data['bridge_type'] == record['bridge_type'])
            ]
            score_col = 'condition_score'
        elif infra_type == 'housing':
            similar_data = data[
                (abs(data['age'] - record['age']) <= 15) &
                (data['climate_zone'] == record['climate_zone']) &
                (data['property_type'] == record['property_type'])
            ]
            score_col = 'condition_rating'
        else:  # road
            similar_data = data[
                (abs(data['age'] - record['age']) <= 10) &
                (data['climate_zone'] == record['climate_zone']) &
                (data['road_type'] == record['road_type'])
            ]
            score_col = 'condition_score'
        
        if len(similar_data) > 1:
            peer_scores = similar_data[score_col]
            percentile = (peer_scores < record[score_col]).mean() * 100
            
            comparison = {
                'peer_group_size': len(similar_data),
                'peer_average_score': float(peer_scores.mean()),
                'current_percentile': float(percentile),
                'performance': 'Above Average' if percentile > 60 else 'Below Average' if percentile < 40 else 'Average'
            }
        else:
            comparison = {
                'peer_group_size': 0,
                'note': 'Insufficient peer data for comparison'
            }
        
        return comparison
    
    def _analyze_costs(self, record) -> Dict[str, Any]:
        """Analyze historical and projected costs"""
        total_historical_cost = record.get('total_maintenance_cost', 0) + record.get('total_renovation_cost', 0)
        
        # Estimate annual cost based on age and condition
        age = record['age']
        annual_maintenance_cost = total_historical_cost / max(age, 1) if total_historical_cost > 0 else 10000
        
        return {
            'total_historical_cost': float(total_historical_cost),
            'estimated_annual_maintenance': float(annual_maintenance_cost),
            'cost_efficiency': 'Good' if annual_maintenance_cost < 15000 else 'Poor',
            'projected_5_year_cost': float(annual_maintenance_cost * 5),
            'cost_per_year_of_service': float(total_historical_cost / max(age, 1))
        }
    
    def _generate_summary(self, record, infra_type) -> str:
        """Generate comprehensive summary"""
        if infra_type == 'bridge':
            score = record['condition_score']
        elif infra_type == 'housing':
            score = record['condition_rating']
        else:
            score = record['condition_score']
        
        age = record['age']
        last_maintenance = record.get('last_maintenance_years', 999)
        
        summary = f"""
INFRASTRUCTURE ANALYSIS SUMMARY
{'='*50}

This {age}-year-old {infra_type} currently has a condition score of {score:.2f}. 

Based on historical data and current condition:
• Age-related deterioration is {"significant" if age > 40 else "moderate" if age > 20 else "minimal"}
• Last maintenance was {last_maintenance} years ago {"(OVERDUE)" if last_maintenance > 10 else ""}
• Climate zone: {record['climate_zone'].title()} - {"Higher" if record['climate_zone'] == 'cold' else "Moderate"} deterioration risk

IMMEDIATE ACTIONS NEEDED:
"""
        
        if score < 3 or last_maintenance > 15:
            summary += "🚨 CRITICAL: Immediate inspection and repairs required\n"
        elif score < 5 or last_maintenance > 10:
            summary += "⚠️  HIGH PRIORITY: Schedule comprehensive maintenance within 6 months\n"
        elif last_maintenance > 7:
            summary += "📋 MEDIUM PRIORITY: Plan maintenance within next 12 months\n"
        else:
            summary += "✅ GOOD CONDITION: Continue regular monitoring schedule\n"
        
        return summary

def main():
    """Main function to demonstrate comprehensive infrastructure analysis"""
    print("🏗️  COMPREHENSIVE INFRASTRUCTURE ANALYSIS SYSTEM")
    print("="*60)
    
    analyzer = InfrastructureAnalyzer()
    
    # Generate comprehensive dataset
    print("📊 Generating comprehensive infrastructure dataset...")
    datasets = analyzer.generate_comprehensive_dataset(n_samples=2000)  # Reduced for demo
    
    # Train models
    analyzer.train_comprehensive_models(datasets)
    
    # Demonstrate analysis for different infrastructure types
    print("\\n" + "="*60)
    print("🔍 DEMONSTRATING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Sample analyses
    infrastructure_samples = [
        ('BR_00100', 'bridge'),
        ('HU_00500', 'housing'),
        ('RD_00300', 'road')
    ]
    
    for infra_id, infra_type in infrastructure_samples:
        print(f"\\n🏗️  ANALYZING {infra_type.upper()}: {infra_id}")
        print("-" * 50)
        
        analysis = analyzer.analyze_infrastructure(infra_id, infra_type)
        
        # Print key information
        print(f"📍 Location: {analysis['basic_info']['location']['climate_zone'].title()} climate zone")
        print(f"🏗️  Built: {analysis['basic_info']['construction_year']} (Age: {analysis['basic_info']['current_age']} years)")
        print(f"📊 Current Condition: {analysis['current_condition']['score']:.2f} - {analysis['current_condition']['category']}")
        print(f"⚠️  Risk Level: {analysis['risk_assessment']['risk_level']}")
        
        print("\\n🔍 Historical Analysis:")
        hist = analysis['historical_analysis']
        print(f"   • Total maintenance events: {hist['total_events']}")
        print(f"   • Trend: {hist['trend']}")
        
        print("\\n📋 Recommendations:")
        for i, rec in enumerate(analysis['maintenance_recommendations'][:2], 1):
            print(f"   {i}. [{rec['priority']}] {rec['action']} - {rec['timeline']}")
        
        print("\\n🔮 Future Projections:")
        proj = analysis['future_projections']['projections']
        print(f"   • 5 years: {proj['5_years']['projected_score']:.1f} ({proj['5_years']['projected_category']})")
        print(f"   • 10 years: {proj['10_years']['projected_score']:.1f} ({proj['10_years']['projected_category']})")
        
        print("\\n" + analysis['summary'])
    
    print("\\n" + "="*60)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*60)
    print("\\n🎯 The system can now:")
    print("   • Analyze any infrastructure with comprehensive historical context")
    print("   • Provide detailed condition assessments and risk analysis")
    print("   • Generate actionable maintenance recommendations")
    print("   • Project future conditions and costs")
    print("   • Compare performance with peer infrastructure")
    print("\\n💡 Usage: analyzer.analyze_infrastructure('INFRA_ID', 'infrastructure_type')")

if __name__ == "__main__":
    main()