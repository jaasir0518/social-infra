#!/usr/bin/env python3
"""
Interactive Bridge Query System
Ask about any bridge condition and get comprehensive analysis
"""

import sys
import os
from pathlib import Path
import json

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge_analysis_system import BridgeConditionAnalyzer

def display_bridge_menu(analyzer):
    """Display available bridges"""
    print("\\n🌉 AVAILABLE BRIDGES:")
    print("-" * 50)
    
    if analyzer.bridge_data is None:
        print("❌ No bridge data available")
        return []
    
    bridge_ids = []
    for _, bridge in analyzer.bridge_data.iterrows():
        condition = bridge['condition_score']
        status = "🟢" if condition >= 7 else "🟡" if condition >= 5.5 else "🟠" if condition >= 4 else "🔴"
        print(f"{status} {bridge['bridge_id']}: {bridge['name']}")
        print(f"   Location: {bridge['location']} | Condition: {condition:.1f}/10 | Age: {bridge['age']} years")
        bridge_ids.append(bridge['bridge_id'])
    
    return bridge_ids

def display_detailed_analysis(analysis):
    """Display comprehensive bridge analysis in readable format"""
    if 'error' in analysis:
        print(f"❌ {analysis['error']}")
        if 'available_bridges' in analysis:
            print("\\nAvailable bridges:")
            for bridge in analysis['available_bridges']:
                print(f"  • {bridge['bridge_id']}: {bridge['name']} (Condition: {bridge['condition_score']:.1f})")
        return
    
    # Executive Summary
    print("\\n" + "="*80)
    print(analysis['executive_summary'])
    
    # Current Condition Details
    current = analysis['current_condition_assessment']
    print(f"\\n🔍 DETAILED CONDITION ASSESSMENT:")
    print(f"   Overall Score: {current['overall_score']:.1f}/10 ({current['condition_category']})")
    print(f"   Status: {current['status_indicator']} {current['implications']}")
    print(f"   Action Required: {current['action_urgency']}")
    
    if current['component_conditions']:
        print(f"\\n   Component Conditions:")
        for component, condition in current['component_conditions'].items():
            flag = "⚠️" if condition['needs_attention'] else "✅"
            print(f"     {flag} {component}: {condition['condition']}")
    
    # Risk Analysis
    risk = analysis['risk_evaluation']
    risk_color = "🔴" if risk['overall_risk_level'] == 'Critical' else "🟠" if risk['overall_risk_level'] == 'High' else "🟡" if risk['overall_risk_level'] == 'Moderate' else "🟢"
    print(f"\\n⚠️  RISK ASSESSMENT:")
    print(f"   Risk Level: {risk_color} {risk['overall_risk_level']} (Score: {risk['risk_score']}/20)")
    print(f"   Description: {risk['risk_description']}")
    
    if risk['identified_risks']:
        print(f"   Identified Risks:")
        for i, risk_item in enumerate(risk['identified_risks'][:3], 1):
            print(f"     {i}. {risk_item}")
    
    # Maintenance Recommendations
    maintenance = analysis['maintenance_recommendations']
    print(f"\\n🔧 MAINTENANCE PLAN:")
    print(f"   Priority Tasks: {maintenance['priority_summary']['critical_tasks']} Critical, {maintenance['priority_summary']['high_priority_tasks']} High Priority")
    print(f"   Estimated Costs: {maintenance['cost_estimate']['total_range']}")
    print(f"   Inspection Frequency: {maintenance['recommended_inspection_frequency']}")
    
    # Show top priority tasks
    priority_tasks = [t for t in maintenance['maintenance_tasks'] if t['priority'] in ['Critical', 'High']]
    if priority_tasks:
        print(f"\\n   Next Actions:")
        for i, task in enumerate(priority_tasks[:3], 1):
            priority_emoji = "🚨" if task['priority'] == 'Critical' else "⚠️"
            print(f"     {i}. {priority_emoji} {task['task']} - {task['timeline']}")
    
    # Future Projections
    future = analysis['future_projections']
    print(f"\\n🔮 CONDITION PROJECTIONS:")
    print(f"   Current: {future['current_condition']:.1f}/10")
    
    for period in ['1_years', '5_years', '10_years']:
        proj = future['projections'][period]
        period_name = period.replace('_years', ' year' if period.startswith('1') else ' years')
        trend = "📈" if proj['projected_condition'] > future['current_condition'] else "📉"
        print(f"   {period_name}: {trend} {proj['projected_condition']:.1f}/10 ({proj['condition_category']}) - {proj['action_needed']}")
    
    # Cost Analysis
    costs = analysis['cost_analysis']
    print(f"\\n💰 FINANCIAL ANALYSIS:")
    print(f"   Annual Maintenance: ${costs['current_costs']['estimated_annual_maintenance']:,}")
    print(f"   Cost Efficiency: {costs['cost_efficiency']['rating']}")
    print(f"   Replacement Cost: ${costs['replacement_analysis']['estimated_replacement_cost']:,}")
    print(f"   Budget Recommendation: ${costs['budget_recommendations']['annual_budget_allocation']:,}/year")
    
    # Historical Context
    history = analysis['historical_analysis']
    print(f"\\n📚 HISTORICAL CONTEXT:")
    print(f"   Construction Era: {history['construction_era']['period']}")
    print(f"   Age Category: {history['age_analysis']['age_category']} ({history['age_analysis']['life_stage_assessment']})")
    print(f"   Performance vs Expected: {history['performance_analysis']['assessment']}")
    print(f"   Maintenance History: {history['maintenance_history_estimate']['maintenance_status']}")

def main():
    """Interactive bridge analysis system"""
    print("🌉 INTERACTIVE BRIDGE CONDITION ANALYSIS SYSTEM")
    print("="*60)
    print("Ask about any bridge infrastructure condition!")
    print("Get comprehensive history, current status, and projections.")
    
    # Initialize the analyzer
    analyzer = BridgeConditionAnalyzer()
    
    # Load data
    print("\\n📊 Loading bridge data...")
    bridge_data = analyzer.load_bridge_data()
    
    if bridge_data is None:
        print("❌ Could not load bridge data. Please check data files.")
        return
    
    print(f"✅ Loaded {len(bridge_data)} bridges successfully!")
    
    while True:
        # Show available bridges
        bridge_ids = display_bridge_menu(analyzer)
        
        print("\\n" + "="*60)
        print("💬 BRIDGE ANALYSIS QUERY")
        print("="*60)
        print("Commands:")
        print("  • Enter bridge ID (e.g., 'B001', 'B002', etc.) for detailed analysis")
        print("  • Type 'summary' for quick overview of all bridges")
        print("  • Type 'exit' to quit")
        
        user_input = input("\\n🔍 Enter your query: ").strip().upper()
        
        if user_input == 'EXIT':
            print("\\n👋 Thank you for using the Bridge Analysis System!")
            break
        
        elif user_input == 'SUMMARY':
            print("\\n📋 BRIDGE FLEET SUMMARY:")
            print("="*60)
            
            if analyzer.bridge_data is not None:
                for _, bridge in analyzer.bridge_data.iterrows():
                    condition = bridge['condition_score']
                    age = bridge['age']
                    status = "🟢" if condition >= 7 else "🟡" if condition >= 5.5 else "🟠" if condition >= 4 else "🔴"
                    
                    print(f"\\n{status} {bridge['bridge_id']}: {bridge['name']}")
                    print(f"   📍 Location: {bridge['location']}")
                    print(f"   📊 Condition: {condition:.1f}/10 | Age: {age} years")
                    print(f"   💰 Annual Cost: ${bridge['estimated_annual_maintenance_cost']:,.0f}")
                    
                    # Quick risk assessment
                    if condition < 6:
                        print(f"   ⚠️  Action: High priority maintenance needed")
                    elif bridge['estimated_years_since_maintenance'] > 10:
                        print(f"   📋 Action: Maintenance overdue")
                    else:
                        print(f"   ✅ Action: Routine monitoring")
        
        elif user_input in bridge_ids:
            print(f"\\n🔍 Analyzing {user_input}...")
            analysis = analyzer.analyze_bridge(user_input)
            display_detailed_analysis(analysis)
        
        else:
            print(f"\\n❌ Bridge '{user_input}' not found.")
            print("Please enter a valid bridge ID from the list above.")
        
        input("\\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()