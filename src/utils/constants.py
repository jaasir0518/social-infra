"""Constants used throughout the social infrastructure prediction system."""

# Model types
BRIDGE_MODEL = "bridge"
HOUSING_MODEL = "housing"
ROAD_MODEL = "road"
UTILITY_MODEL = "utility"
ENSEMBLE_MODEL = "ensemble"

MODEL_TYPES = [BRIDGE_MODEL, HOUSING_MODEL, ROAD_MODEL, UTILITY_MODEL, ENSEMBLE_MODEL]

# Infrastructure categories
INFRASTRUCTURE_CATEGORIES = {
    "bridge": ["highway", "pedestrian", "railway", "overpass"],
    "housing": ["residential", "commercial", "mixed_use", "affordable"],
    "road": ["highway", "arterial", "collector", "local"],
    "utility": ["water", "electricity", "gas", "sewage", "telecommunications"]
}

# Data quality thresholds
MIN_DATA_QUALITY_SCORE = 0.7
MAX_MISSING_VALUES_RATIO = 0.3
MIN_SAMPLE_SIZE = 100

# Geographic constants
EARTH_RADIUS_KM = 6371.0
METERS_PER_DEGREE_LAT = 111000.0
DEFAULT_BUFFER_DISTANCE = 1000  # meters
DEFAULT_GRID_SIZE = 500  # meters

# Time constants
SECONDS_PER_DAY = 86400
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365
DEFAULT_FORECAST_HORIZON = 12  # months

# Model performance thresholds
MIN_MODEL_ACCURACY = 0.75
MIN_MODEL_PRECISION = 0.70
MIN_MODEL_RECALL = 0.70
MIN_MODEL_F1_SCORE = 0.70

# File formats
SUPPORTED_DATA_FORMATS = [".csv", ".xlsx", ".xls", ".json", ".parquet", ".feather"]
SUPPORTED_MODEL_FORMATS = [".pkl", ".joblib", ".h5", ".onnx"]
SUPPORTED_CONFIG_FORMATS = [".yaml", ".yml", ".json"]

# API constants
DEFAULT_API_TIMEOUT = 30  # seconds
MAX_BATCH_SIZE = 1000
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500

# Feature engineering constants
OUTLIER_THRESHOLD_STD = 3.0  # Standard deviations
MIN_FEATURE_IMPORTANCE = 0.01
MAX_CORRELATION_THRESHOLD = 0.95

# Data validation constants
MAX_CATEGORICAL_UNIQUE_VALUES = 100
MIN_NUMERIC_RANGE = 1e-10
MAX_STRING_LENGTH = 1000

# Infrastructure condition scales
CONDITION_SCALES = {
    "excellent": 1.0,
    "good": 0.8,
    "fair": 0.6,
    "poor": 0.4,
    "critical": 0.2,
    "failed": 0.0
}

# Priority levels
PRIORITY_LEVELS = {
    "critical": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
    "deferred": 5
}

# Default model hyperparameters
DEFAULT_HYPERPARAMETERS = {
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    },
    "lightgbm": {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "random_state": 42
    },
    "neural_network": {
        "hidden_layer_sizes": (100, 50),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": 0.001,
        "max_iter": 1000,
        "random_state": 42
    }
}

# Logging levels
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Status codes
STATUS_CODES = {
    "SUCCESS": 200,
    "CREATED": 201,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "INTERNAL_ERROR": 500
}

# Color schemes for visualization
COLOR_SCHEMES = {
    "infrastructure": {
        "bridge": "#FF6B6B",
        "housing": "#4ECDC4", 
        "road": "#45B7D1",
        "utility": "#96CEB4"
    },
    "condition": {
        "excellent": "#2ECC71",
        "good": "#F1C40F",
        "fair": "#E67E22",
        "poor": "#E74C3C",
        "critical": "#8E44AD",
        "failed": "#34495E"
    },
    "priority": {
        "critical": "#E74C3C",
        "high": "#E67E22",
        "medium": "#F1C40F",
        "low": "#2ECC71",
        "deferred": "#95A5A6"
    }
}