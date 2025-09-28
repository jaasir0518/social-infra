#!/usr/bin/env python3
"""
NTAD Bridge System Usage Guide
Complete guide for using the enhanced bridge analysis system with NTAD data
"""

def print_usage_guide():
    print("🌉 NTAD-ENHANCED BRIDGE ANALYSIS SYSTEM - USAGE GUIDE")
    print("="*80)
    print("Transform bridge management with National Bridge Inventory data & ML models")
    
    print("\\n📋 SYSTEM OVERVIEW")
    print("-"*50)
    print("This system integrates:")
    print("✅ NTAD National Bridge Inventory dataset (10,000+ bridges)")
    print("✅ Advanced machine learning models (7 algorithms × 5 targets)")
    print("✅ Your local bridge inventory data") 
    print("✅ Comprehensive condition analysis and predictions")
    
    print("\\n🚀 QUICK START")
    print("-"*50)
    
    print("\\n1. CREATE NTAD DATASET:")
    print("   python ntad_bridge_integration.py")
    print("   📊 Creates comprehensive dataset with 10,000 bridges")
    print("   📁 Saves to: data/ntad/nbi_processed_data.csv")
    print("   ⏱️  Time: ~30 seconds")
    
    print("\\n2. TRAIN ML MODELS:")
    print("   python enhanced_bridge_trainer.py")
    print("   🤖 Trains 35+ models for condition prediction")
    print("   📁 Saves models to: models/ directory")
    print("   ⏱️  Time: ~5-10 minutes")
    
    print("\\n3. ANALYZE BRIDGES:")
    print("   python ask_bridge.py B001")
    print("   🔍 Get comprehensive analysis with NTAD insights")
    print("   📊 Compare with national benchmarks")
    print("   ⏱️  Time: ~5 seconds per bridge")
    
    print("\\n📊 AVAILABLE ANALYSES")
    print("-"*50)
    
    analyses = [
        ("Bridge Condition Assessment", "Overall, deck, superstructure, substructure conditions"),
        ("Risk Evaluation", "Age, traffic, material, environmental risk factors"),
        ("Maintenance Planning", "Prioritized tasks with timelines and costs"),
        ("Future Projections", "1, 5, 10-year condition forecasts"),
        ("Cost Analysis", "Annual maintenance, replacement economics"),
        ("National Benchmarking", "Compare with similar bridges nationwide"),
        ("NTAD Model Predictions", "ML-based condition predictions"),
        ("Executive Reports", "Professional summaries for decision-makers")
    ]
    
    for analysis, description in analyses:
        print(f"   ✅ {analysis}: {description}")
    
    print("\\n🎯 BRIDGE QUERY COMMANDS")
    print("-"*50)
    
    commands = [
        ("python ask_bridge.py B001", "Analyze Main Street Bridge"),
        ("python ask_bridge.py B002", "Analyze River Crossing Bridge"),
        ("python ask_bridge.py B005", "Analyze Railway Bridge (Critical)"),
        ("python demo_bridge_queries.py", "Full system demonstration"),
        ("python ntad_enhanced_analyzer.py", "Enhanced analysis with NTAD models"),
        ("python bridge_analysis_system.py", "Complete fleet analysis")
    ]
    
    for command, description in commands:
        print(f"   {command}")
        print(f"   → {description}")
        print()
    
    print("\\n📈 MODEL CAPABILITIES")
    print("-"*50)
    
    targets = [
        "Overall Condition Score (1-10 scale)",
        "Deck Condition (4-9 scale)",  
        "Superstructure Condition (4-9 scale)",
        "Substructure Condition (4-9 scale)",
        "Maintenance Priority Score (0-20 scale)"
    ]
    
    algorithms = [
        "Random Forest Regressor",
        "Gradient Boosting Regressor", 
        "Extra Trees Regressor",
        "Ridge Regression",
        "Elastic Net Regression",
        "Support Vector Regression",
        "Multi-layer Perceptron"
    ]
    
    print("🎯 PREDICTION TARGETS:")
    for target in targets:
        print(f"   • {target}")
    
    print("\\n🤖 ML ALGORITHMS:")
    for algorithm in algorithms:
        print(f"   • {algorithm}")
    
    print("\\n📊 DATASET FEATURES")
    print("-"*50)
    
    feature_categories = {
        "Structural": ["Age", "Length", "Width", "Area", "Material", "Skew"],
        "Traffic": ["Daily Traffic", "Traffic per Lane", "Truck Percentage"],
        "Environmental": ["Weather Risk", "Seismic Risk", "Scour Risk"],
        "Economic": ["Improvement Cost", "Cost per Area", "Budget Impact"],
        "Operational": ["Fracture Critical", "Inspection Frequency", "Overdue Status"],
        "Geographic": ["Interstate Status", "Urban/Rural", "Functional Class"]
    }
    
    for category, features in feature_categories.items():
        print(f"\\n🏷️  {category.upper()} FEATURES:")
        for feature in features:
            print(f"   • {feature}")
    
    print("\\n💡 USAGE SCENARIOS")
    print("-"*50)
    
    scenarios = [
        ("Bridge Inspection Planning", "Identify bridges needing immediate attention"),
        ("Maintenance Budget Planning", "Estimate costs and prioritize work"),
        ("Asset Management", "Track bridge portfolio health over time"),
        ("Risk Assessment", "Evaluate safety and operational risks"),
        ("Benchmarking", "Compare performance with similar bridges"),
        ("Reporting", "Generate executive summaries and reports"),
        ("Predictive Maintenance", "Forecast when bridges will need work"),
        ("Resource Allocation", "Optimize maintenance crew assignments")
    ]
    
    for scenario, description in scenarios:
        print(f"\\n🎯 {scenario.upper()}:")
        print(f"   {description}")
        
        if scenario == "Bridge Inspection Planning":
            print("   Example: python ask_bridge.py B005")
            print("   → Railway Bridge shows critical condition, needs immediate inspection")
        elif scenario == "Maintenance Budget Planning": 
            print("   Example: python bridge_analysis_system.py")
            print("   → Fleet analysis shows $370K annual maintenance needs")
        elif scenario == "Predictive Maintenance":
            print("   Example: Bridge B001 projected to reach 'Fair' condition in 5 years")
    
    print("\\n📁 FILE STRUCTURE")
    print("-"*50)
    
    files = [
        ("data/ntad/", "NTAD dataset files (10,000 bridges)"),
        ("data/raw/bridges/", "Local bridge inventory CSV files"),
        ("models/", "Trained ML models and preprocessors"),
        ("ask_bridge.py", "Quick bridge analysis script"),
        ("bridge_analysis_system.py", "Comprehensive fleet analysis"),
        ("ntad_bridge_integration.py", "NTAD data processing"),
        ("enhanced_bridge_trainer.py", "ML model training"),
        ("ntad_enhanced_analyzer.py", "Enhanced analysis with NTAD models")
    ]
    
    for file_path, description in files:
        print(f"   📁 {file_path}: {description}")
    
    print("\\n🔧 TROUBLESHOOTING")
    print("-"*50)
    
    issues = [
        ("Import errors", "Run scripts from the social-infra directory"),
        ("No NTAD data", "Run: python ntad_bridge_integration.py"),
        ("No trained models", "Run: python enhanced_bridge_trainer.py"),
        ("Bridge not found", "Check available bridges with ask_bridge.py"),
        ("Performance issues", "Models are CPU-intensive, use powerful hardware"),
        ("Missing dependencies", "Install: pandas, numpy, scikit-learn, joblib")
    ]
    
    for issue, solution in issues:
        print(f"   ❌ {issue}: {solution}")
    
    print("\\n✅ SUCCESS METRICS")
    print("-"*50)
    
    metrics = [
        "Model R² scores > 0.80 for condition predictions",
        "Cross-validation performance within 5% of test performance", 
        "Prediction agreement with actual conditions > 85%",
        "Analysis time < 10 seconds per bridge",
        "Fleet analysis completed in < 2 minutes",
        "Executive reports generated automatically"
    ]
    
    for metric in metrics:
        print(f"   ✅ {metric}")
    
    print("\\n🎯 NEXT LEVEL FEATURES")
    print("-"*50)
    
    advanced = [
        "Real-time data integration from bridge sensors",
        "Automated alert systems for critical conditions",
        "Web dashboard for bridge fleet management",
        "Mobile app for field inspections",
        "Integration with GIS systems",
        "Advanced visualization and mapping",
        "Custom model training for specific bridge types",
        "Multi-year budget optimization"
    ]
    
    for feature in advanced:
        print(f"   🚀 {feature}")
    
    print("\\n" + "="*80)
    print("🌉 READY TO REVOLUTIONIZE BRIDGE MANAGEMENT!")
    print("="*80)
    print("Start with: python ask_bridge.py B005")
    print("Your bridges deserve data-driven care! 🌉")

if __name__ == "__main__":
    print_usage_guide()