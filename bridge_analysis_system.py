#!/usr/bin/env python3
"""
Real Bridge Data Analysis System (Small Dataset Version)
Analyzes bridge conditions using actual data files without requiring large training sets
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class BridgeConditionAnalyzer:
    """
    Bridge condition analyzer optimized for small datasets using rule-based analysis
    combined with data-driven insights from available bridge data
    """
    
    def __init__(self, data_path: str = "data/raw/bridges"):
        self.data_path = Path(data_path)
        self.bridge_data = None
        self.condition_thresholds = {
            'excellent': 8.5,
            'good': 7.0,
            'fair': 5.5,
            'poor': 4.0,
            'critical': 2.5
        }
        
    def load_bridge_data(self):
        """Load and enhance bridge data from CSV files"""
        print("🌉 Loading bridge data from files...")
        
        try:
            # Load bridge inventory
            inventory_file = self.data_path / "bridge_inventory.csv"
            conditions_file = self.data_path / "bridge_conditions.csv"
            
            if not inventory_file.exists() or not conditions_file.exists():
                print(f"❌ Bridge data files not found in {self.data_path}")
                return None
            
            # Read the data
            inventory_df = pd.read_csv(inventory_file, comment='#')
            conditions_df = pd.read_csv(conditions_file, comment='#')
            
            print(f"✅ Loaded {len(inventory_df)} bridge inventory records")
            print(f"✅ Loaded {len(conditions_df)} bridge condition records")
            
            # Merge the datasets
            bridge_data = inventory_df.merge(conditions_df, on='bridge_id', how='inner')
            
            # Enhance with calculated features
            bridge_data = self._enhance_data(bridge_data)
            
            print(f"📊 Enhanced dataset: {len(bridge_data)} bridges with {len(bridge_data.columns)} features")
            
            self.bridge_data = bridge_data
            return bridge_data
            
        except Exception as e:
            print(f"❌ Error loading bridge data: {e}")
            return None
    
    def _enhance_data(self, data):
        """Enhance bridge data with calculated features"""
        enhanced_data = data.copy()
        
        # Calculate age
        current_year = 2024
        enhanced_data['age'] = current_year - enhanced_data['construction_year']
        
        # Calculate bridge area
        enhanced_data['bridge_area_sqm'] = enhanced_data['length_m'] * enhanced_data['width_m']
        
        # Estimated traffic volume (realistic estimates based on location)
        location_traffic_map = {
            'Downtown': 45000,
            'North District': 25000,
            'Industrial Zone': 35000,
            'Park Area': 5000,
            'Central Station': 30000
        }
        
        enhanced_data['estimated_daily_traffic'] = enhanced_data['location'].map(
            lambda x: location_traffic_map.get(x, 20000)
        )
        
        # Climate zone based on location (simplified)
        enhanced_data['climate_zone'] = 'temperate'  # Assuming all bridges in temperate zone
        
        # Estimated last maintenance (based on condition and age)
        # Better condition suggests more recent maintenance
        condition_factor = enhanced_data['condition_score'] / 10
        age_factor = enhanced_data['age']
        
        # Estimate years since maintenance
        enhanced_data['estimated_years_since_maintenance'] = np.maximum(
            1, age_factor * (1.5 - condition_factor)
        ).round().astype(int)
        
        # Risk categories based on age and condition
        enhanced_data['age_risk'] = pd.cut(
            enhanced_data['age'], 
            bins=[0, 20, 40, 100], 
            labels=['low', 'medium', 'high']
        )
        
        enhanced_data['condition_risk'] = pd.cut(
            enhanced_data['condition_score'],
            bins=[0, 4, 6, 8, 10],
            labels=['critical', 'high', 'medium', 'low']
        )
        
        # Estimated maintenance cost based on bridge size and age
        enhanced_data['estimated_annual_maintenance_cost'] = (
            enhanced_data['bridge_area_sqm'] * 25 +  # Base cost per sqm
            enhanced_data['age'] * 500 +  # Age factor
            (10 - enhanced_data['condition_score']) * 2000  # Condition factor
        )
        
        # Priority score (higher = more urgent)
        enhanced_data['maintenance_priority_score'] = (
            (10 - enhanced_data['condition_score']) * 3 +  # Condition urgency
            (enhanced_data['age'] / 10) * 2 +  # Age factor
            (enhanced_data['estimated_years_since_maintenance'] / 5) * 2  # Maintenance overdue
        )
        
        return enhanced_data
    
    def analyze_bridge(self, bridge_id: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a specific bridge using rule-based assessment
        """
        if self.bridge_data is None:
            return {"error": "No bridge data loaded"}
        
        # Find the bridge
        bridge_record = self.bridge_data[self.bridge_data['bridge_id'] == bridge_id]
        
        if bridge_record.empty:
            # Show available bridges if not found
            available_bridges = self.bridge_data[['bridge_id', 'name', 'condition_score']].to_dict('records')
            return {
                "error": f"Bridge {bridge_id} not found",
                "available_bridges": available_bridges
            }
        
        bridge = bridge_record.iloc[0]
        
        # Generate comprehensive analysis
        analysis = {
            'bridge_id': bridge_id,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'basic_information': self._get_bridge_basics(bridge),
            'current_condition_assessment': self._assess_current_condition(bridge),
            'historical_analysis': self._analyze_bridge_history(bridge),
            'risk_evaluation': self._evaluate_risks(bridge),
            'maintenance_recommendations': self._generate_maintenance_plan(bridge),
            'cost_analysis': self._analyze_costs(bridge),
            'future_projections': self._project_condition(bridge),
            'comparative_analysis': self._compare_with_fleet(bridge),
            'detailed_inspection_report': self._generate_inspection_report(bridge),
            'action_plan': self._create_action_plan(bridge),
            'executive_summary': self._create_executive_summary(bridge)
        }
        
        return analysis
    
    def _get_bridge_basics(self, bridge) -> Dict[str, Any]:
        """Extract basic bridge information"""
        return {
            'name': bridge['name'],
            'location': bridge['location'],
            'coordinates': {
                'latitude': float(bridge['latitude']),
                'longitude': float(bridge['longitude'])
            },
            'construction_details': {
                'year': int(bridge['construction_year']),
                'age_years': int(bridge['age']),
                'material': bridge['material']
            },
            'physical_characteristics': {
                'length_m': float(bridge['length_m']),
                'width_m': float(bridge['width_m']),
                'total_area_sqm': float(bridge['bridge_area_sqm'])
            },
            'operational_data': {
                'estimated_daily_traffic': int(bridge['estimated_daily_traffic']),
                'climate_zone': bridge['climate_zone']
            }
        }
    
    def _assess_current_condition(self, bridge) -> Dict[str, Any]:
        """Comprehensive current condition assessment"""
        condition_score = bridge['condition_score']
        overall_rating = bridge['overall_rating']
        
        # Determine condition category and implications
        if condition_score >= self.condition_thresholds['excellent']:
            category = 'Excellent'
            status_color = '🟢'
            implications = 'Bridge is in outstanding condition with minimal maintenance needs'
            action_urgency = 'Routine monitoring sufficient'
        elif condition_score >= self.condition_thresholds['good']:
            category = 'Good'
            status_color = '🔵'
            implications = 'Bridge is in good working condition with normal wear patterns'
            action_urgency = 'Standard maintenance schedule'
        elif condition_score >= self.condition_thresholds['fair']:
            category = 'Fair'
            status_color = '🟡'
            implications = 'Bridge shows moderate wear and requires increased attention'
            action_urgency = 'Enhanced monitoring and preventive maintenance'
        elif condition_score >= self.condition_thresholds['poor']:
            category = 'Poor'
            status_color = '🟠'
            implications = 'Bridge requires significant maintenance and close monitoring'
            action_urgency = 'Major maintenance program needed'
        else:
            category = 'Critical'
            status_color = '🔴'
            implications = 'Bridge condition poses potential safety and operational concerns'
            action_urgency = 'Immediate assessment and intervention required'
        
        # Component assessment
        component_conditions = {}
        components = ['deck_condition', 'superstructure_condition', 'substructure_condition']
        
        for component in components:
            if component in bridge.index:
                comp_name = component.replace('_condition', '').title()
                comp_value = bridge[component]
                component_conditions[comp_name] = {
                    'condition': comp_value,
                    'needs_attention': comp_value.lower() in ['poor', 'fair']
                }
        
        return {
            'overall_score': float(condition_score),
            'overall_rating': float(overall_rating),
            'condition_category': category,
            'status_indicator': status_color,
            'implications': implications,
            'action_urgency': action_urgency,
            'component_conditions': component_conditions,
            'maintenance_flag': bool(bridge['maintenance_needed']),
            'last_inspection_date': bridge.get('inspection_date', 'Not recorded')
        }
    
    def _analyze_bridge_history(self, bridge) -> Dict[str, Any]:
        """Analyze bridge historical context and performance"""
        age = bridge['age']
        construction_year = bridge['construction_year']
        condition_score = bridge['condition_score']
        
        # Construction era analysis
        if construction_year < 1960:
            era = "Pre-Interstate Era"
            era_characteristics = "Built before modern design standards"
        elif construction_year < 1990:
            era = "Interstate Era" 
            era_characteristics = "Built during major highway expansion"
        elif construction_year < 2010:
            era = "Modern Era"
            era_characteristics = "Built with contemporary design standards"
        else:
            era = "Contemporary Era"
            era_characteristics = "Recently constructed with latest standards"
        
        # Performance analysis
        expected_condition = 9.5 - (age * 0.08)  # Expected deterioration curve
        performance_vs_expected = condition_score - expected_condition
        
        if performance_vs_expected > 1.5:
            performance_assessment = "Significantly above expected"
            performance_note = "Excellent maintenance history or exceptional construction"
        elif performance_vs_expected > 0.5:
            performance_assessment = "Above expected"
            performance_note = "Good maintenance and care"
        elif performance_vs_expected > -0.5:
            performance_assessment = "As expected"
            performance_note = "Normal wear and maintenance patterns"
        elif performance_vs_expected > -1.5:
            performance_assessment = "Below expected"
            performance_note = "May indicate deferred maintenance or harsh conditions"
        else:
            performance_assessment = "Significantly below expected"
            performance_note = "Requires investigation into maintenance history and conditions"
        
        # Maintenance timeline estimation
        estimated_maintenance_years = bridge['estimated_years_since_maintenance']
        
        return {
            'construction_era': {
                'period': era,
                'characteristics': era_characteristics
            },
            'age_analysis': {
                'current_age': int(age),
                'age_category': 'New' if age < 15 else 'Mature' if age < 40 else 'Aging',
                'life_stage_assessment': self._get_life_stage_assessment(age)
            },
            'performance_analysis': {
                'expected_condition': float(expected_condition),
                'actual_condition': float(condition_score),
                'performance_gap': float(performance_vs_expected),
                'assessment': performance_assessment,
                'notes': performance_note
            },
            'maintenance_history_estimate': {
                'estimated_years_since_maintenance': int(estimated_maintenance_years),
                'maintenance_status': 'Overdue' if estimated_maintenance_years > 10 else 'Current',
                'estimated_total_maintenance_cycles': max(1, age // 8)
            }
        }
    
    def _evaluate_risks(self, bridge) -> Dict[str, Any]:
        """Comprehensive risk evaluation"""
        risks = []
        risk_factors = {}
        total_risk_score = 0
        
        # Age-based risks
        age = bridge['age']
        if age > 50:
            risks.append("High age-related structural fatigue risk")
            risk_factors['age'] = 'high'
            total_risk_score += 4
        elif age > 30:
            risks.append("Moderate age-related deterioration")
            risk_factors['age'] = 'medium'
            total_risk_score += 2
        else:
            risk_factors['age'] = 'low'
            total_risk_score += 1
        
        # Condition-based risks
        condition = bridge['condition_score']
        if condition < 5:
            risks.append("Structural integrity concerns requiring immediate attention")
            risk_factors['condition'] = 'critical'
            total_risk_score += 5
        elif condition < 6.5:
            risks.append("Accelerating deterioration patterns")
            risk_factors['condition'] = 'high'
            total_risk_score += 3
        elif condition < 7.5:
            risks.append("Normal wear requiring monitoring")
            risk_factors['condition'] = 'medium'
            total_risk_score += 2
        else:
            risk_factors['condition'] = 'low'
            total_risk_score += 1
        
        # Traffic-based risks
        traffic = bridge['estimated_daily_traffic']
        if traffic > 40000:
            risks.append("Very high traffic loading causing accelerated wear")
            risk_factors['traffic'] = 'high'
            total_risk_score += 3
        elif traffic > 20000:
            risks.append("Moderate to high traffic loading")
            risk_factors['traffic'] = 'medium'
            total_risk_score += 2
        else:
            risk_factors['traffic'] = 'low'
            total_risk_score += 1
        
        # Material-based risks
        material = bridge['material'].lower()
        if 'steel' in material:
            risks.append("Steel corrosion potential requiring monitoring")
            risk_factors['material'] = 'medium'
            total_risk_score += 2
        elif 'wood' in material:
            risks.append("Wood deterioration and pest damage risk")
            risk_factors['material'] = 'medium'
            total_risk_score += 2
        else:
            risk_factors['material'] = 'low'
            total_risk_score += 1
        
        # Maintenance-based risks
        years_since_maintenance = bridge['estimated_years_since_maintenance']
        if years_since_maintenance > 12:
            risks.append("Significantly overdue for major maintenance")
            risk_factors['maintenance'] = 'critical'
            total_risk_score += 4
        elif years_since_maintenance > 8:
            risks.append("Overdue for scheduled maintenance")
            risk_factors['maintenance'] = 'high'
            total_risk_score += 3
        elif years_since_maintenance > 5:
            risks.append("Approaching maintenance interval")
            risk_factors['maintenance'] = 'medium'
            total_risk_score += 2
        else:
            risk_factors['maintenance'] = 'low'
            total_risk_score += 1
        
        # Overall risk assessment
        if total_risk_score >= 15:
            overall_risk = "Critical"
            risk_description = "Multiple high-risk factors require immediate comprehensive action"
        elif total_risk_score >= 12:
            overall_risk = "High"
            risk_description = "Significant risk factors require priority attention"
        elif total_risk_score >= 8:
            overall_risk = "Moderate"
            risk_description = "Some risk factors require monitoring and planned action"
        else:
            overall_risk = "Low"
            risk_description = "Minimal risk factors, routine management sufficient"
        
        return {
            'overall_risk_level': overall_risk,
            'risk_score': total_risk_score,
            'risk_description': risk_description,
            'identified_risks': risks,
            'risk_factor_breakdown': risk_factors,
            'risk_mitigation_urgency': 'Immediate' if total_risk_score >= 15 else 'High' if total_risk_score >= 12 else 'Medium'
        }
    
    def _generate_maintenance_plan(self, bridge) -> Dict[str, Any]:
        """Generate comprehensive maintenance plan"""
        condition = bridge['condition_score']
        age = bridge['age']
        years_since_maintenance = bridge['estimated_years_since_maintenance']
        priority_score = bridge['maintenance_priority_score']
        
        maintenance_tasks = []
        
        # Immediate tasks based on condition
        if condition < 4:
            maintenance_tasks.append({
                'task': 'Emergency structural assessment',
                'priority': 'Critical',
                'timeline': '1-2 weeks',
                'estimated_cost': '$15,000 - $30,000',
                'description': 'Immediate professional structural evaluation'
            })
            maintenance_tasks.append({
                'task': 'Load restriction evaluation',
                'priority': 'Critical', 
                'timeline': '2-4 weeks',
                'estimated_cost': '$5,000 - $15,000',
                'description': 'Assess if traffic restrictions needed'
            })
        
        elif condition < 6:
            maintenance_tasks.append({
                'task': 'Comprehensive bridge inspection',
                'priority': 'High',
                'timeline': '1-3 months',
                'estimated_cost': '$8,000 - $20,000',
                'description': 'Detailed assessment of all bridge components'
            })
            maintenance_tasks.append({
                'task': 'Major rehabilitation planning',
                'priority': 'High',
                'timeline': '3-6 months',
                'estimated_cost': '$50,000 - $200,000',
                'description': 'Plan and execute major repair work'
            })
        
        # Age-based maintenance
        if age > 40:
            maintenance_tasks.append({
                'task': 'Fatigue crack inspection',
                'priority': 'High',
                'timeline': '6 months',
                'estimated_cost': '$10,000 - $25,000',
                'description': 'Specialized inspection for age-related fatigue'
            })
        
        # Material-specific maintenance
        if 'steel' in bridge['material'].lower():
            maintenance_tasks.append({
                'task': 'Corrosion protection renewal',
                'priority': 'Medium',
                'timeline': '1 year',
                'estimated_cost': '$20,000 - $60,000',
                'description': 'Anti-corrosion treatment and protective coating'
            })
        
        # Maintenance schedule based on timeline
        if years_since_maintenance > 10:
            maintenance_tasks.append({
                'task': 'Overdue major maintenance',
                'priority': 'High',
                'timeline': '6 months',
                'estimated_cost': '$75,000 - $300,000',
                'description': 'Comprehensive maintenance addressing deferred work'
            })
        
        # Component-specific maintenance
        components = ['deck_condition', 'superstructure_condition', 'substructure_condition']
        for component in components:
            if component in bridge.index:
                comp_condition = bridge[component].lower()
                comp_name = component.replace('_condition', '').title()
                
                if comp_condition in ['poor', 'fair']:
                    maintenance_tasks.append({
                        'task': f'{comp_name} repair and rehabilitation',
                        'priority': 'High' if comp_condition == 'poor' else 'Medium',
                        'timeline': '6-12 months',
                        'estimated_cost': '$25,000 - $100,000',
                        'description': f'Address {comp_name.lower()} condition issues'
                    })
        
        # Default routine maintenance if no specific issues
        if not maintenance_tasks or all(task['priority'] == 'Medium' for task in maintenance_tasks):
            maintenance_tasks.append({
                'task': 'Routine preventive maintenance',
                'priority': 'Low',
                'timeline': 'Annual',
                'estimated_cost': '$5,000 - $15,000',
                'description': 'Regular cleaning, minor repairs, and inspection'
            })
        
        # Calculate total estimated cost
        cost_ranges = []
        for task in maintenance_tasks:
            cost_str = task['estimated_cost']
            if ' - ' in cost_str:
                low, high = cost_str.replace('$', '').replace(',', '').split(' - ')
                cost_ranges.append((int(low), int(high)))
        
        total_low = sum([c[0] for c in cost_ranges])
        total_high = sum([c[1] for c in cost_ranges])
        
        return {
            'maintenance_tasks': maintenance_tasks,
            'priority_summary': {
                'critical_tasks': len([t for t in maintenance_tasks if t['priority'] == 'Critical']),
                'high_priority_tasks': len([t for t in maintenance_tasks if t['priority'] == 'High']),
                'total_tasks': len(maintenance_tasks)
            },
            'cost_estimate': {
                'total_range': f'${total_low:,} - ${total_high:,}',
                'immediate_needs': f'${sum([c[0] for t, c in zip(maintenance_tasks, cost_ranges) if t["priority"] in ["Critical", "High"]]):,}',
                'annual_budget_recommendation': f'${int((total_low + total_high) / 2 / 5):,}'  # Spread over 5 years
            },
            'maintenance_priority_score': float(priority_score),
            'recommended_inspection_frequency': 'Monthly' if condition < 5 else 'Quarterly' if condition < 7 else 'Annually'
        }
    
    def _analyze_costs(self, bridge) -> Dict[str, Any]:
        """Comprehensive cost analysis"""
        annual_maintenance_cost = bridge['estimated_annual_maintenance_cost']
        bridge_area = bridge['bridge_area_sqm']
        age = bridge['age']
        
        # Calculate various cost metrics
        cost_per_sqm_annual = annual_maintenance_cost / bridge_area
        lifetime_cost = annual_maintenance_cost * age
        
        # Replacement cost estimate (rough)
        replacement_cost_per_sqm = 2500  # Modern bridge construction cost
        replacement_cost = bridge_area * replacement_cost_per_sqm
        
        # Cost efficiency assessment
        if cost_per_sqm_annual < 30:
            cost_efficiency = "Excellent"
        elif cost_per_sqm_annual < 50:
            cost_efficiency = "Good"
        elif cost_per_sqm_annual < 75:
            cost_efficiency = "Fair"
        else:
            cost_efficiency = "Poor"
        
        # Future cost projections
        condition_factor = (10 - bridge['condition_score']) / 10
        age_factor = 1 + (age / 100)
        
        projected_costs = {}
        for years in [1, 3, 5, 10]:
            base_cost = annual_maintenance_cost * years
            escalation = 1 + (0.03 * years)  # 3% annual inflation
            condition_multiplier = 1 + (condition_factor * years * 0.1)
            
            projected_cost = base_cost * escalation * condition_multiplier
            projected_costs[f'{years}_year'] = int(projected_cost)
        
        return {
            'current_costs': {
                'estimated_annual_maintenance': int(annual_maintenance_cost),
                'cost_per_square_meter': int(cost_per_sqm_annual),
                'lifetime_maintenance_cost': int(lifetime_cost)
            },
            'cost_efficiency': {
                'rating': cost_efficiency,
                'benchmark_comparison': 'Below average' if cost_per_sqm_annual > 60 else 'Above average'
            },
            'replacement_analysis': {
                'estimated_replacement_cost': int(replacement_cost),
                'cost_to_replace_ratio': f'{(lifetime_cost / replacement_cost):.2f}',
                'replacement_recommendation': 'Consider' if lifetime_cost / replacement_cost > 0.7 else 'Continue maintenance'
            },
            'projected_costs': projected_costs,
            'budget_recommendations': {
                'immediate_reserve_needed': int(annual_maintenance_cost * 2),
                'annual_budget_allocation': int(annual_maintenance_cost * 1.2),
                'long_term_capital_reserve': int(replacement_cost * 0.05)
            }
        }
    
    def _project_condition(self, bridge) -> Dict[str, Any]:
        """Project future bridge condition"""
        current_condition = bridge['condition_score']
        age = bridge['age']
        
        # Base deterioration rate (points per year)
        base_deterioration = 0.08
        
        # Adjust deterioration rate based on factors
        traffic_factor = 1.0
        if bridge['estimated_daily_traffic'] > 30000:
            traffic_factor = 1.3
        elif bridge['estimated_daily_traffic'] > 15000:
            traffic_factor = 1.1
        
        material_factor = 1.0
        if 'steel' in bridge['material'].lower():
            material_factor = 1.1
        
        age_factor = 1.0
        if age > 40:
            age_factor = 1.2
        elif age > 25:
            age_factor = 1.1
        
        adjusted_deterioration = base_deterioration * traffic_factor * material_factor * age_factor
        
        # Project future conditions
        projections = {}
        for years in [1, 3, 5, 10, 15]:
            # Calculate deterioration
            deteriorated_condition = current_condition - (adjusted_deterioration * years)
            
            # Factor in maintenance cycles (assume major maintenance every 8-10 years)
            maintenance_cycles = years // 9
            if maintenance_cycles > 0:
                # Each maintenance cycle improves condition by 1.5-2.5 points
                maintenance_improvement = maintenance_cycles * 2.0
                deteriorated_condition += maintenance_improvement
            
            # Keep within reasonable bounds
            deteriorated_condition = max(1.0, min(10.0, deteriorated_condition))
            
            # Determine projected category
            if deteriorated_condition >= 7.5:
                category = 'Good'
                action_needed = 'Routine maintenance'
            elif deteriorated_condition >= 6.0:
                category = 'Fair'
                action_needed = 'Increased monitoring'
            elif deteriorated_condition >= 4.0:
                category = 'Poor'
                action_needed = 'Major maintenance required'
            else:
                category = 'Critical'
                action_needed = 'Replacement consideration'
            
            projections[f'{years}_years'] = {
                'projected_condition': round(deteriorated_condition, 1),
                'condition_category': category,
                'action_needed': action_needed,
                'maintenance_cycles_assumed': maintenance_cycles
            }
        
        return {
            'current_condition': float(current_condition),
            'deterioration_rate_per_year': round(adjusted_deterioration, 3),
            'projections': projections,
            'factors_considered': {
                'traffic_impact': traffic_factor,
                'material_impact': material_factor,
                'age_impact': age_factor
            },
            'key_assumptions': [
                'Regular maintenance cycles every 8-10 years',
                'Current usage patterns continue',
                'No major external factors (disasters, policy changes)'
            ]
        }
    
    def _compare_with_fleet(self, bridge) -> Dict[str, Any]:
        """Compare bridge with others in the dataset"""
        if len(self.bridge_data) <= 1:
            return {'note': 'Insufficient data for fleet comparison'}
        
        # Get fleet statistics
        fleet_conditions = self.bridge_data['condition_score']
        fleet_ages = self.bridge_data['age']
        
        # Bridge's position in fleet
        condition_percentile = (fleet_conditions < bridge['condition_score']).mean() * 100
        age_percentile = (fleet_ages < bridge['age']).mean() * 100
        
        # Find similar bridges (same material, similar age ±10 years)
        similar_bridges = self.bridge_data[
            (self.bridge_data['material'] == bridge['material']) &
            (abs(self.bridge_data['age'] - bridge['age']) <= 10) &
            (self.bridge_data['bridge_id'] != bridge['bridge_id'])
        ]
        
        comparison_result = {
            'fleet_statistics': {
                'total_bridges_in_fleet': len(self.bridge_data),
                'fleet_average_condition': float(fleet_conditions.mean()),
                'fleet_average_age': float(fleet_ages.mean()),
                'condition_percentile': float(condition_percentile),
                'age_percentile': float(age_percentile)
            },
            'performance_ranking': {
                'condition_rank': f"{int(condition_percentile)}th percentile",
                'relative_performance': 'Above average' if condition_percentile > 60 else 'Below average' if condition_percentile < 40 else 'Average'
            }
        }
        
        if len(similar_bridges) > 0:
            similar_avg_condition = similar_bridges['condition_score'].mean()
            comparison_result['similar_bridges_comparison'] = {
                'similar_bridge_count': len(similar_bridges),
                'similar_bridges_avg_condition': float(similar_avg_condition),
                'performance_vs_similar': 'Better' if bridge['condition_score'] > similar_avg_condition else 'Worse',
                'condition_difference': float(bridge['condition_score'] - similar_avg_condition)
            }
        
        return comparison_result
    
    def _generate_inspection_report(self, bridge) -> Dict[str, Any]:
        """Generate detailed inspection report format"""
        return {
            'inspection_summary': {
                'bridge_id': bridge['bridge_id'],
                'inspection_type': 'Comprehensive Condition Assessment',
                'inspector_notes': 'Data-driven analysis based on available bridge records',
                'weather_conditions': 'Data analysis (weather independent)',
                'access_conditions': 'Full access assumed'
            },
            'structural_assessment': {
                'overall_rating': float(bridge['condition_score']),
                'primary_concerns': self._identify_primary_concerns(bridge),
                'secondary_observations': self._identify_secondary_observations(bridge)
            },
            'component_ratings': self._get_component_ratings(bridge),
            'recommended_follow_up': {
                'next_inspection_date': self._calculate_next_inspection_date(bridge),
                'special_inspections_needed': self._identify_special_inspections(bridge),
                'monitoring_requirements': self._identify_monitoring_needs(bridge)
            }
        }
    
    def _create_action_plan(self, bridge) -> Dict[str, Any]:
        """Create prioritized action plan"""
        condition = bridge['condition_score']
        
        actions = []
        
        # Immediate actions (0-30 days)
        if condition < 5:
            actions.append({
                'timeframe': 'Immediate (0-30 days)',
                'actions': [
                    'Conduct emergency structural assessment',
                    'Evaluate need for load restrictions',
                    'Increase inspection frequency to monthly'
                ],
                'responsibility': 'Bridge Engineer/Department Head',
                'budget_impact': 'High'
            })
        
        # Short-term actions (1-6 months)
        short_term_actions = ['Schedule comprehensive inspection']
        if bridge['estimated_years_since_maintenance'] > 8:
            short_term_actions.append('Plan major maintenance program')
        if condition < 7:
            short_term_actions.append('Develop rehabilitation plan')
        
        actions.append({
            'timeframe': 'Short-term (1-6 months)',
            'actions': short_term_actions,
            'responsibility': 'Maintenance Team/Engineering Consultant',
            'budget_impact': 'Medium to High'
        })
        
        # Long-term actions (6 months - 2 years)
        long_term_actions = ['Implement maintenance plan']
        if bridge['age'] > 30:
            long_term_actions.append('Develop long-term asset management strategy')
        
        actions.append({
            'timeframe': 'Long-term (6 months - 2 years)',
            'actions': long_term_actions,
            'responsibility': 'Asset Management Team',
            'budget_impact': 'Medium'
        })
        
        return {
            'priority_actions': actions,
            'success_metrics': [
                'Condition score improvement or stabilization',
                'Reduced maintenance frequency needs',
                'Cost efficiency improvements',
                'Public safety maintained'
            ],
            'risk_mitigation': 'Action plan addresses primary risk factors identified in analysis'
        }
    
    def _create_executive_summary(self, bridge) -> str:
        """Create executive summary report"""
        condition = bridge['condition_score']
        age = bridge['age']
        name = bridge['name']
        location = bridge['location']
        
        # Determine status emoji and priority
        if condition >= 7.5:
            status_emoji = "🟢"
            priority = "Low Priority"
        elif condition >= 6.0:
            status_emoji = "🟡"
            priority = "Medium Priority"
        elif condition >= 4.0:
            status_emoji = "🟠"
            priority = "High Priority"
        else:
            status_emoji = "🔴"
            priority = "Critical Priority"
        
        summary = f"""
{status_emoji} EXECUTIVE SUMMARY: {name.upper()}
{'='*60}

BRIDGE IDENTIFICATION:
• Name: {name}
• Location: {location}
• Bridge ID: {bridge['bridge_id']}
• Construction Year: {bridge['construction_year']} (Age: {age} years)

CURRENT STATUS:
• Overall Condition: {condition:.1f}/10.0 ({self._get_condition_category(condition)})
• Priority Level: {priority}
• Maintenance Status: {'⚠️ Overdue' if bridge['estimated_years_since_maintenance'] > 10 else '✅ Current'}

KEY FINDINGS:
• Bridge is {age} years old, built in {bridge['construction_year']}
• Current condition score of {condition:.1f} {'exceeds' if condition > 7 else 'meets' if condition > 5.5 else 'falls below'} acceptable standards
• Estimated {bridge['estimated_years_since_maintenance']} years since major maintenance
• Annual maintenance cost estimated at ${bridge['estimated_annual_maintenance_cost']:,.0f}

IMMEDIATE ACTIONS REQUIRED:
"""
        
        if condition < 4:
            summary += "🚨 CRITICAL: Immediate structural assessment and potential load restrictions\n"
        elif condition < 6:
            summary += "⚠️ HIGH PRIORITY: Schedule comprehensive maintenance within 6 months\n"
        elif bridge['estimated_years_since_maintenance'] > 10:
            summary += "📋 MEDIUM PRIORITY: Address overdue maintenance within 12 months\n"
        else:
            summary += "✅ ROUTINE: Continue regular monitoring and maintenance schedule\n"
        
        summary += f"""
FINANCIAL IMPLICATIONS:
• Estimated immediate needs: ${bridge['estimated_annual_maintenance_cost'] * 2:,.0f}
• 5-year maintenance projection: ${bridge['estimated_annual_maintenance_cost'] * 5 * 1.2:,.0f}
• Replacement cost estimate: ${bridge['bridge_area_sqm'] * 2500:,.0f}

RECOMMENDATION:
"""
        
        if condition < 5:
            summary += f"Bridge requires immediate attention and should be prioritized in emergency maintenance budget. Consider load restrictions pending detailed assessment."
        elif condition < 7:
            summary += f"Bridge should be included in next fiscal year's major maintenance program to prevent further deterioration."
        else:
            summary += f"Bridge is in acceptable condition. Continue routine maintenance and monitoring."
        
        summary += f"""

Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis confidence: High (based on available data and industry standards)
"""
        
        return summary
    
    def _get_condition_category(self, condition):
        """Get condition category from score"""
        if condition >= 8.5:
            return "Excellent"
        elif condition >= 7.0:
            return "Good"
        elif condition >= 5.5:
            return "Fair"
        elif condition >= 4.0:
            return "Poor"
        else:
            return "Critical"
    
    def _get_life_stage_assessment(self, age):
        """Get life stage assessment"""
        if age < 10:
            return "New - Minimal deterioration expected"
        elif age < 25:
            return "Mature - Normal wear patterns"
        elif age < 50:
            return "Aging - Increased maintenance needs"
        else:
            return "Elderly - Intensive care required"
    
    def _identify_primary_concerns(self, bridge):
        """Identify primary structural concerns"""
        concerns = []
        condition = bridge['condition_score']
        age = bridge['age']
        
        if condition < 5:
            concerns.append("Overall structural condition below acceptable standards")
        
        if age > 50:
            concerns.append("Age-related deterioration and fatigue concerns")
        
        if bridge['estimated_years_since_maintenance'] > 12:
            concerns.append("Significantly overdue for major maintenance")
        
        if 'steel' in bridge['material'].lower() and condition < 7:
            concerns.append("Potential corrosion issues in steel components")
        
        return concerns if concerns else ["No major structural concerns identified"]
    
    def _identify_secondary_observations(self, bridge):
        """Identify secondary observations"""
        observations = []
        
        if bridge['estimated_daily_traffic'] > 30000:
            observations.append("High traffic loading may accelerate wear")
        
        if bridge['age'] > 30:
            observations.append("Bridge approaching major maintenance lifecycle")
        
        observations.append(f"Bridge area of {bridge['bridge_area_sqm']:.0f} sqm requires regular maintenance")
        
        return observations
    
    def _get_component_ratings(self, bridge):
        """Get component ratings"""
        components = {}
        component_fields = ['deck_condition', 'superstructure_condition', 'substructure_condition']
        
        for field in component_fields:
            if field in bridge.index:
                component_name = field.replace('_condition', '').title()
                components[component_name] = bridge[field]
        
        return components
    
    def _calculate_next_inspection_date(self, bridge):
        """Calculate next inspection date"""
        condition = bridge['condition_score']
        
        if condition < 5:
            months = 3  # Quarterly for critical bridges
        elif condition < 7:
            months = 6  # Semi-annually for fair bridges
        else:
            months = 12  # Annually for good bridges
        
        next_date = datetime.now() + timedelta(days=months*30)
        return next_date.strftime('%Y-%m-%d')
    
    def _identify_special_inspections(self, bridge):
        """Identify special inspection needs"""
        inspections = []
        
        if bridge['age'] > 40:
            inspections.append("Fatigue and fracture critical member inspection")
        
        if 'steel' in bridge['material'].lower():
            inspections.append("Corrosion assessment")
        
        if bridge['condition_score'] < 6:
            inspections.append("Load capacity evaluation")
        
        return inspections if inspections else ["No special inspections required at this time"]
    
    def _identify_monitoring_needs(self, bridge):
        """Identify monitoring requirements"""
        needs = []
        
        if bridge['condition_score'] < 5:
            needs.append("Monthly visual inspections")
        
        if bridge['estimated_daily_traffic'] > 30000:
            needs.append("Traffic load monitoring")
        
        if bridge['age'] > 40:
            needs.append("Structural health monitoring system consideration")
        
        return needs if needs else ["Standard inspection schedule sufficient"]

def main():
    """Main function to demonstrate the bridge analysis system"""
    print("🌉 COMPREHENSIVE BRIDGE CONDITION ANALYSIS SYSTEM")
    print("="*70)
    
    # Initialize analyzer
    analyzer = BridgeConditionAnalyzer()
    
    # Load bridge data
    bridge_data = analyzer.load_bridge_data()
    
    if bridge_data is None:
        print("❌ Failed to load bridge data. Please check data files.")
        return
    
    print("\\n" + "="*70)
    print("📋 AVAILABLE BRIDGES FOR ANALYSIS")
    print("="*70)
    
    # Show all available bridges
    for _, bridge in bridge_data.iterrows():
        condition_color = "🟢" if bridge['condition_score'] >= 7 else "🟡" if bridge['condition_score'] >= 5.5 else "🟠" if bridge['condition_score'] >= 4 else "🔴"
        print(f"{condition_color} {bridge['bridge_id']}: {bridge['name']} - Condition: {bridge['condition_score']:.1f}/10")
    
    print("\\n" + "="*70)
    print("🔍 DETAILED BRIDGE ANALYSIS")
    print("="*70)
    
    # Analyze each bridge in detail
    for bridge_id in bridge_data['bridge_id'].tolist():
        print(f"\\n{'🌉 ' + '='*15} ANALYZING BRIDGE {bridge_id} {'='*15}")
        
        # Get comprehensive analysis
        analysis = analyzer.analyze_bridge(bridge_id)
        
        if 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
            continue
        
        # Display executive summary
        print(analysis['executive_summary'])
        
        # Display key metrics in a structured format
        basic_info = analysis['basic_information']
        current_condition = analysis['current_condition_assessment']
        risk_eval = analysis['risk_evaluation']
        maintenance = analysis['maintenance_recommendations']
        
        print(f"\\n📊 QUICK REFERENCE METRICS:")
        print(f"   • Age: {basic_info['construction_details']['age_years']} years")
        print(f"   • Material: {basic_info['construction_details']['material']}")
        print(f"   • Size: {basic_info['physical_characteristics']['length_m']:.1f}m × {basic_info['physical_characteristics']['width_m']:.1f}m")
        print(f"   • Traffic: {basic_info['operational_data']['estimated_daily_traffic']:,} vehicles/day")
        print(f"   • Risk Level: {risk_eval['overall_risk_level']}")
        print(f"   • Priority Tasks: {maintenance['priority_summary']['critical_tasks'] + maintenance['priority_summary']['high_priority_tasks']}")
        print(f"   • Est. Annual Cost: ${analysis['cost_analysis']['current_costs']['estimated_annual_maintenance']:,}")
        
        # Show immediate action items
        critical_tasks = [task for task in maintenance['maintenance_tasks'] if task['priority'] in ['Critical', 'High']]
        if critical_tasks:
            print(f"\\n⚠️  IMMEDIATE ACTIONS NEEDED:")
            for task in critical_tasks[:3]:  # Show top 3 priority tasks
                print(f"   • [{task['priority']}] {task['task']} - {task['timeline']}")
        
        # Show future condition projection
        future = analysis['future_projections']
        print(f"\\n🔮 CONDITION PROJECTIONS:")
        print(f"   • 5-year: {future['projections']['5_years']['projected_condition']} ({future['projections']['5_years']['condition_category']})")
        print(f"   • 10-year: {future['projections']['10_years']['projected_condition']} ({future['projections']['10_years']['condition_category']})")
    
    print("\\n" + "="*70)
    print("✅ COMPREHENSIVE BRIDGE ANALYSIS COMPLETE")  
    print("="*70)
    print("\\n🎯 SYSTEM CAPABILITIES DEMONSTRATED:")
    print("   ✓ Real bridge data integration from CSV files")
    print("   ✓ Comprehensive condition assessment and risk analysis")
    print("   ✓ Historical context and performance evaluation")
    print("   ✓ Detailed maintenance planning and cost analysis")
    print("   ✓ Future condition projections")
    print("   ✓ Executive summary reports")
    print("   ✓ Action plans with timelines and priorities")
    print("\\n💡 ASK ABOUT ANY BRIDGE:")
    print("   Simply provide a bridge ID (B001, B002, etc.) and get instant")
    print("   comprehensive analysis including history, current condition,")
    print("   risks, costs, recommendations, and future projections!")

if __name__ == "__main__":
    main()