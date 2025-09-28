"""
Settings and configuration management for the social infrastructure prediction system.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, ConfigDict
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field, ConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    model_config = ConfigDict(extra='ignore')
    
    url: str = Field(default="postgresql://localhost:5432/social_infra", env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")


class APISettings(BaseSettings):
    """API configuration settings."""
    model_config = ConfigDict(extra='ignore')
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, env="API_RATE_LIMIT_RPM")
    burst_size: int = Field(default=10, env="API_RATE_LIMIT_BURST")


class ModelSettings(BaseSettings):
    """Model configuration settings."""
    model_config = ConfigDict(extra='ignore')
    
    # Bridge prediction model
    bridge_algorithm: str = Field(default="xgboost", env="BRIDGE_ALGORITHM")
    bridge_params: Dict[str, Any] = Field(default={
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8
    })
    
    # Housing prediction model
    housing_algorithm: str = Field(default="random_forest", env="HOUSING_ALGORITHM")
    housing_params: Dict[str, Any] = Field(default={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    })
    
    # Road prediction model
    road_algorithm: str = Field(default="lightgbm", env="ROAD_ALGORITHM")
    road_params: Dict[str, Any] = Field(default={
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8
    })
    
    # Ensemble settings
    ensemble_method: str = Field(default="voting", env="ENSEMBLE_METHOD")
    ensemble_weights: List[float] = Field(default=[0.3, 0.3, 0.4])


class DataSettings(BaseSettings):
    """Data configuration settings."""
    model_config = ConfigDict(extra='ignore')
    
    raw_data_path: str = Field(default="data/raw", env="RAW_DATA_PATH")
    processed_data_path: str = Field(default="data/processed", env="PROCESSED_DATA_PATH")
    interim_data_path: str = Field(default="data/interim", env="INTERIM_DATA_PATH")
    
    # Training settings
    test_size: float = Field(default=0.2, env="TEST_SIZE")
    validation_size: float = Field(default=0.1, env="VALIDATION_SIZE")
    random_state: int = Field(default=42, env="RANDOM_STATE")
    
    # Cross-validation
    cv_folds: int = Field(default=5, env="CV_FOLDS")
    cv_stratified: bool = Field(default=True, env="CV_STRATIFIED")


class FeatureSettings(BaseSettings):
    """Feature engineering configuration settings."""
    model_config = ConfigDict(extra='ignore')
    
    # Spatial features
    spatial_buffer_distance: int = Field(default=1000, env="SPATIAL_BUFFER_DISTANCE")
    spatial_grid_size: int = Field(default=500, env="SPATIAL_GRID_SIZE")
    
    # Temporal features
    temporal_window_size: int = Field(default=12, env="TEMPORAL_WINDOW_SIZE")
    temporal_seasonality: bool = Field(default=True, env="TEMPORAL_SEASONALITY")
    
    # Infrastructure metrics
    age_weight: float = Field(default=0.3, env="AGE_WEIGHT")
    condition_weight: float = Field(default=0.4, env="CONDITION_WEIGHT")
    usage_weight: float = Field(default=0.3, env="USAGE_WEIGHT")


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    model_config = ConfigDict(extra='ignore')
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        env="LOG_FORMAT"
    )
    rotation: str = Field(default="10 MB", env="LOG_ROTATION")
    retention: str = Field(default="30 days", env="LOG_RETENTION")


class MonitoringSettings(BaseSettings):
    """Monitoring configuration settings."""
    model_config = ConfigDict(extra='ignore')
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="social_infrastructure_prediction", env="MLFLOW_EXPERIMENT_NAME")
    
    # Weights & Biases
    wandb_project: str = Field(default="social-infrastructure", env="WANDB_PROJECT")
    wandb_entity: Optional[str] = Field(default=None, env="WANDB_ENTITY")


class DeploymentSettings(BaseSettings):
    """Deployment configuration settings."""
    model_config = ConfigDict(extra='ignore')
    
    # Docker
    docker_image_name: str = Field(default="social-infra-prediction", env="DOCKER_IMAGE_NAME")
    docker_tag: str = Field(default="latest", env="DOCKER_TAG")
    
    # Kubernetes
    k8s_namespace: str = Field(default="social-infra", env="K8S_NAMESPACE")
    k8s_replicas: int = Field(default=3, env="K8S_REPLICAS")
    
    # Resources
    cpu_request: str = Field(default="500m", env="CPU_REQUEST")
    memory_request: str = Field(default="1Gi", env="MEMORY_REQUEST")
    cpu_limit: str = Field(default="2", env="CPU_LIMIT")
    memory_limit: str = Field(default="4Gi", env="MEMORY_LIMIT")


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections."""
    model_config = ConfigDict(extra='ignore', env_file='.env', env_file_encoding='utf-8')
    
    # Project info
    project_name: str = Field(default="social_infrastructure_prediction", env="PROJECT_NAME")
    project_version: str = Field(default="0.1.0", env="PROJECT_VERSION")
    project_description: str = Field(
        default="ML system for predicting infrastructure maintenance needs",
        env="PROJECT_DESCRIPTION"
    )
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    models: ModelSettings = ModelSettings()
    data: DataSettings = DataSettings()
    features: FeatureSettings = FeatureSettings()
    logging: LoggingSettings = LoggingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    deployment: DeploymentSettings = DeploymentSettings()
    
    @classmethod
    def load_from_yaml(cls, config_path: str = "config.yaml") -> "Settings":
        """Load settings from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save_to_yaml(self, config_path: str = "config.yaml") -> None:
        """Save settings to YAML file."""
        config_data = self.dict()
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory path."""
        return self.project_root / "data"
    
    @property
    def models_dir(self) -> Path:
        """Get the models directory path."""
        return self.project_root / "models"
    
    @property
    def logs_dir(self) -> Path:
        """Get the logs directory path."""
        return self.project_root / "monitoring" / "logs"


# Global settings instance
settings = Settings()

# Try to load from YAML if available
try:
    settings = Settings.load_from_yaml()
except FileNotFoundError:
    # Use default settings if config.yaml doesn't exist
    pass