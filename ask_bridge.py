#!/usr/bin/env python3
"""
Simple Bridge Query - Ask about any bridge by ID
Usage: python ask_bridge.py B001
"""

import sys
import os

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge_analysis_system import BridgeConditionAnalyzer

def ask_about_bridge(bridge_id):
    """Ask about a specific bridge and get comprehensive analysis"""
    
    # Initialize the system
    analyzer = BridgeConditionAnalyzer()
    
    # Load the data
    bridge_data = analyzer.load_bridge_data()
    
    if bridge_data is None:
        print("❌ Could not load bridge data from CSV files")
        return
    
    # Get the analysis
    analysis = analyzer.analyze_bridge(bridge_id)
    
    if 'error' in analysis:
        print(f"❌ {analysis['error']}")
        
        # Show available bridges
        if 'available_bridges' in analysis:
            print("\\n🌉 Available bridges:")
            for bridge in analysis['available_bridges']:
                print(f"  • {bridge['bridge_id']}: {bridge['name']} (Condition: {bridge['condition_score']:.1f}/10)")
        return
    
    # Display the executive summary (most important info)
    print(analysis['executive_summary'])
    
    # Quick action items
    maintenance = analysis['maintenance_recommendations']
    priority_tasks = [t for t in maintenance['maintenance_tasks'] if t['priority'] in ['Critical', 'High']]
    
    if priority_tasks:
        print("\\n🚨 IMMEDIATE ACTION ITEMS:")
        for i, task in enumerate(priority_tasks[:3], 1):
            priority_emoji = "🚨" if task['priority'] == 'Critical' else "⚠️"
            print(f"   {i}. {priority_emoji} {task['task']} - {task['timeline']}")
            print(f"      Cost: {task['estimated_cost']}")

def main():
    """Main function"""
    
    if len(sys.argv) != 2:
        print("🌉 BRIDGE QUERY SYSTEM")
        print("Usage: python ask_bridge.py <bridge_id>")
        print("\\nExample: python ask_bridge.py B001")
        print("\\nAvailable bridges:")
        
        # Show available bridges
        analyzer = BridgeConditionAnalyzer()
        bridge_data = analyzer.load_bridge_data()
        
        if bridge_data is not None:
            for _, bridge in bridge_data.iterrows():
                condition = bridge['condition_score']
                status = "🟢" if condition >= 7 else "🟡" if condition >= 5.5 else "🟠"
                print(f"  {status} {bridge['bridge_id']}: {bridge['name']} - {condition:.1f}/10")
        
        return
    
    bridge_id = sys.argv[1].upper()
    ask_about_bridge(bridge_id)

if __name__ == "__main__":
    main()