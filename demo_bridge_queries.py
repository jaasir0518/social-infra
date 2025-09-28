#!/usr/bin/env python3
"""
Bridge Query Demo - Shows how to ask about any bridge condition
"""

import sys
import os
import json

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge_analysis_system import BridgeConditionAnalyzer

def demonstrate_bridge_queries():
    """Demonstrate various bridge analysis queries"""
    print("🌉 BRIDGE CONDITION QUERY DEMONSTRATION")
    print("="*70)
    print("This system can analyze any bridge infrastructure condition")
    print("and provide comprehensive history, status, and projections.")
    
    # Initialize analyzer
    analyzer = BridgeConditionAnalyzer()
    
    # Load bridge data
    print("\\n📊 Loading bridge data...")
    bridge_data = analyzer.load_bridge_data()
    
    if bridge_data is None:
        print("❌ Could not load bridge data")
        return
    
    print(f"✅ Loaded {len(bridge_data)} bridges from CSV files")
    
    # Show all available bridges
    print("\\n🌉 AVAILABLE BRIDGES IN SYSTEM:")
    print("-" * 50)
    for _, bridge in bridge_data.iterrows():
        condition = bridge['condition_score']
        status = "🟢" if condition >= 7 else "🟡" if condition >= 5.5 else "🟠" if condition >= 4 else "🔴"
        print(f"{status} {bridge['bridge_id']}: {bridge['name']} - {condition:.1f}/10")
    
    # Demonstrate different query types
    queries = [
        ("B001", "Main Street Bridge - Good condition bridge analysis"),
        ("B005", "Railway Bridge - Aging bridge requiring attention"),
        ("B002", "River Crossing Bridge - Fair condition analysis"),
    ]
    
    for bridge_id, description in queries:
        print(f"\\n{'='*70}")
        print(f"🔍 QUERY EXAMPLE: Analyzing {bridge_id} ({description})")
        print("="*70)
        
        # Get analysis
        analysis = analyzer.analyze_bridge(bridge_id)
        
        if 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
            continue
        
        # Show key information
        basic_info = analysis['basic_information']
        condition_assessment = analysis['current_condition_assessment']
        risk_eval = analysis['risk_evaluation']
        maintenance_plan = analysis['maintenance_recommendations']
        future_proj = analysis['future_projections']
        cost_analysis = analysis['cost_analysis']
        
        print(f"\\n📍 BRIDGE: {basic_info['name']} ({bridge_id})")
        print(f"   Location: {basic_info['location']}")
        print(f"   Built: {basic_info['construction_details']['year']} (Age: {basic_info['construction_details']['age_years']} years)")
        print(f"   Size: {basic_info['physical_characteristics']['length_m']:.1f}m × {basic_info['physical_characteristics']['width_m']:.1f}m")
        print(f"   Material: {basic_info['construction_details']['material']}")
        
        print(f"\\n🔍 CURRENT CONDITION:")
        print(f"   Overall Score: {condition_assessment['overall_score']:.1f}/10.0 ({condition_assessment['condition_category']})")
        print(f"   Status: {condition_assessment['status_indicator']} {condition_assessment['implications']}")
        print(f"   Maintenance Flag: {'⚠️ Required' if condition_assessment['maintenance_flag'] else '✅ Not urgent'}")
        
        print(f"\\n⚠️  RISK ASSESSMENT:")
        print(f"   Risk Level: {risk_eval['overall_risk_level']} (Score: {risk_eval['risk_score']}/20)")
        print(f"   Description: {risk_eval['risk_description']}")
        if risk_eval['identified_risks']:
            print(f"   Key Risks:")
            for i, risk in enumerate(risk_eval['identified_risks'][:2], 1):
                print(f"     {i}. {risk}")
        
        print(f"\\n🔧 MAINTENANCE RECOMMENDATIONS:")
        print(f"   Priority Tasks: {maintenance_plan['priority_summary']['critical_tasks']} Critical, {maintenance_plan['priority_summary']['high_priority_tasks']} High")
        print(f"   Cost Range: {maintenance_plan['cost_estimate']['total_range']}")
        print(f"   Inspection: {maintenance_plan['recommended_inspection_frequency']}")
        
        # Show top 2 priority tasks
        priority_tasks = [t for t in maintenance_plan['maintenance_tasks'] if t['priority'] in ['Critical', 'High']]
        if priority_tasks:
            print(f"   Next Actions:")
            for i, task in enumerate(priority_tasks[:2], 1):
                print(f"     {i}. [{task['priority']}] {task['task']} - {task['timeline']}")
        
        print(f"\\n🔮 FUTURE PROJECTIONS:")
        current_condition = future_proj['current_condition']
        five_year = future_proj['projections']['5_years']
        ten_year = future_proj['projections']['10_years']
        
        print(f"   Current: {current_condition:.1f}/10")
        print(f"   5-year projection: {five_year['projected_condition']:.1f}/10 ({five_year['condition_category']})")
        print(f"   10-year projection: {ten_year['projected_condition']:.1f}/10 ({ten_year['condition_category']})")
        print(f"   Deterioration rate: {future_proj['deterioration_rate_per_year']:.3f} points/year")
        
        print(f"\\n💰 COST ANALYSIS:")
        print(f"   Annual maintenance: ${cost_analysis['current_costs']['estimated_annual_maintenance']:,}")
        print(f"   Cost efficiency: {cost_analysis['cost_efficiency']['rating']}")
        print(f"   Replacement cost: ${cost_analysis['replacement_analysis']['estimated_replacement_cost']:,}")
        print(f"   Budget recommendation: ${cost_analysis['budget_recommendations']['annual_budget_allocation']:,}/year")
        
        # Show historical context
        history = analysis['historical_analysis']
        print(f"\\n📚 HISTORICAL CONTEXT:")
        print(f"   Era: {history['construction_era']['period']}")
        print(f"   Performance: {history['performance_analysis']['assessment']}")
        print(f"   Maintenance: Est. {history['maintenance_history_estimate']['estimated_years_since_maintenance']} years since major work")
    
    # Show fleet comparison
    print(f"\\n{'='*70}")
    print("📊 FLEET SUMMARY AND COMPARISON")
    print("="*70)
    
    # Calculate fleet statistics
    conditions = bridge_data['condition_score']
    ages = bridge_data['age']
    costs = bridge_data['estimated_annual_maintenance_cost']
    
    print(f"\\n🌉 BRIDGE FLEET OVERVIEW:")
    print(f"   Total Bridges: {len(bridge_data)}")
    print(f"   Average Condition: {conditions.mean():.1f}/10")
    print(f"   Average Age: {ages.mean():.0f} years")
    print(f"   Total Annual Maintenance: ${costs.sum():,.0f}")
    print(f"   Average Annual Maintenance: ${costs.mean():,.0f}")
    
    # Show by condition category
    excellent = len(bridge_data[bridge_data['condition_score'] >= 8.5])
    good = len(bridge_data[(bridge_data['condition_score'] >= 7) & (bridge_data['condition_score'] < 8.5)])
    fair = len(bridge_data[(bridge_data['condition_score'] >= 5.5) & (bridge_data['condition_score'] < 7)])
    poor = len(bridge_data[bridge_data['condition_score'] < 5.5])
    
    print(f"\\n📈 CONDITION DISTRIBUTION:")
    print(f"   🟢 Excellent (8.5+): {excellent} bridges")
    print(f"   🔵 Good (7.0-8.4): {good} bridges")
    print(f"   🟡 Fair (5.5-6.9): {fair} bridges")
    print(f"   🟠 Poor (<5.5): {poor} bridges")
    
    # Priority recommendations
    critical_bridges = bridge_data[bridge_data['condition_score'] < 6]
    print(f"\\n🎯 PRIORITY RECOMMENDATIONS:")
    if len(critical_bridges) > 0:
        print(f"   ⚠️  {len(critical_bridges)} bridges need priority attention:")
        for _, bridge in critical_bridges.iterrows():
            print(f"      • {bridge['bridge_id']}: {bridge['name']} (Condition: {bridge['condition_score']:.1f})")
    else:
        print(f"   ✅ All bridges are in acceptable condition (≥6.0)")
    
    # Show system capabilities
    print(f"\\n{'='*70}")
    print("💡 SYSTEM CAPABILITIES SUMMARY")
    print("="*70)
    print("\\nThis bridge analysis system provides:")
    print("✓ Real-time condition assessment using actual bridge data")
    print("✓ Historical context and performance analysis")
    print("✓ Risk evaluation with multiple factor consideration")  
    print("✓ Detailed maintenance planning with cost estimates")
    print("✓ Future condition projections (1, 5, 10+ years)")
    print("✓ Comparative fleet analysis and benchmarking")
    print("✓ Executive summary reports for decision-making")
    print("✓ Prioritized action plans with timelines")
    
    print("\\n🎯 TO USE THE SYSTEM:")
    print("Simply ask about any bridge by ID (B001, B002, etc.) and get:")
    print("• Complete infrastructure history and timeline")
    print("• Current condition with detailed component assessment")
    print("• Risk factors and safety considerations")
    print("• Maintenance recommendations with cost estimates")
    print("• Future projections and lifecycle planning")
    print("• Comparative analysis with similar infrastructure")
    
    print(f"\\n✅ BRIDGE ANALYSIS SYSTEM READY FOR QUERIES!")

if __name__ == "__main__":
    demonstrate_bridge_queries()