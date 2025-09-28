# Social Infrastructure Prediction

A comprehensive machine learning system for predicting infrastructure maintenance needs and investment priorities across bridges, housing, roads, and utilities.

## Overview

This project uses advanced machine learning techniques to analyze various infrastructure datasets and predict maintenance needs, deterioration patterns, and optimal investment strategies for social infrastructure.

## Features

- **Multi-Infrastructure Modeling**: Specialized models for bridges, housing, roads, and utilities
- **Geospatial Analysis**: Location-based feature engineering and visualization
- **Ensemble Predictions**: Combined model approach for improved accuracy
- **Real-time API**: REST API for predictions and data access
- **Interactive Dashboard**: Web-based visualization and reporting
- **Automated Pipeline**: End-to-end data processing and model training

## Project Structure

```
social_infrastructure_prediction/
├── data/                    # Raw, processed, and interim data
├── src/                     # Source code modules
├── notebooks/               # Jupyter notebooks for analysis
├── models/                  # Trained models and artifacts
├── tests/                   # Test modules
├── scripts/                 # Utility and processing scripts
├── docs/                    # Documentation
├── reports/                 # Generated reports and visualizations
├── api/                     # REST API implementation
├── deployment/              # Docker and Kubernetes configs
└── monitoring/              # Logging and metrics
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml with your settings
   ```

3. **Run Data Pipeline**
   ```bash
   python scripts/preprocessing/run_data_cleaning.py
   python scripts/preprocessing/run_feature_engineering.py
   ```

4. **Train Models**
   ```bash
   python scripts/training/train_models.py
   ```

5. **Start API Server**
   ```bash
   python api/app.py
   ```

## Data Sources

- Bridge inventory and condition data
- Housing board records and property data
- Road network and traffic information
- Utility system data (water, electricity, sewage)
- Demographic and economic indicators
- Weather and satellite imagery
- Government budget allocations

## Models

- **Bridge Prediction**: Deterioration and maintenance scheduling
- **Housing Prediction**: Market trends and maintenance needs
- **Road Prediction**: Traffic patterns and infrastructure wear
- **Ensemble Model**: Combined predictions for comprehensive planning

## API Endpoints

- `GET /api/predict/bridge/{bridge_id}` - Bridge condition prediction
- `GET /api/predict/housing/{area_id}` - Housing market prediction
- `GET /api/predict/road/{road_id}` - Road condition prediction
- `POST /api/predict/batch` - Batch predictions
- `GET /api/data/summary` - Data summary statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

For questions and support, please open an issue on GitHub.