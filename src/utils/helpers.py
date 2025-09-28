"""
Helper utilities for the social infrastructure prediction system.
"""

import os
import sys
import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def setup_logging():
    """Set up basic logging configuration."""
    import logging
    
    # Configure basic logging if loguru is not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', mode='a')
        ]
    )


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of the created directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    ensure_directory(config_path.parent)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def save_model(model: Any, model_path: Union[str, Path]) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        model_path: Path to save the model
    """
    model_path = Path(model_path)
    ensure_directory(model_path.parent)
    
    if model_path.suffix == '.pkl':
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        # Try to use the model's built-in save method
        if hasattr(model, 'save'):
            model.save(str(model_path))
        elif hasattr(model, 'save_model'):
            model.save_model(str(model_path))
        else:
            # Fallback to pickle
            with open(model_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(model, f)


def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model object
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_path.suffix == '.pkl':
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported model file format: {model_path.suffix}")


def generate_hash(data: Union[str, bytes, Dict, List]) -> str:
    """
    Generate SHA-256 hash for data.
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True)
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage statistics
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent(),       # Memory percentage
    }


def timing_context():
    """
    Context manager for timing code execution.
    
    Usage:
        with timing_context() as timer:
            # code to time
            pass
        print(f"Execution time: {timer.elapsed:.4f} seconds")
    """
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None
        
        def __enter__(self):
            self.start_time = datetime.now()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = datetime.now()
            self.elapsed = (self.end_time - self.start_time).total_seconds()
    
    return Timer()


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> bool:
    """
    Validate a DataFrame structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def split_train_test(
    data: Union[pd.DataFrame, np.ndarray],
    target: Optional[Union[pd.Series, np.ndarray]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple[Any, ...]:
    """
    Split data into training and testing sets.
    
    Args:
        data: Input features
        target: Target variable (optional)
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        stratify: Whether to stratify the split
        
    Returns:
        Tuple of split data (X_train, X_test, y_train, y_test) or (X_train, X_test)
    """
    from sklearn.model_selection import train_test_split
    
    if target is not None:
        stratify_param = target if stratify else None
        return train_test_split(
            data, target,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
    else:
        return train_test_split(
            data,
            test_size=test_size,
            random_state=random_state
        )


def calculate_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    method: str = 'haversine'
) -> float:
    """
    Calculate distance between two geographic points.
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
        method: Distance calculation method ('haversine' or 'euclidean')
        
    Returns:
        Distance in meters
    """
    if method == 'haversine':
        # Haversine formula for great-circle distance
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (np.sin(dlat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    elif method == 'euclidean':
        # Simple Euclidean distance (approximation)
        deg_to_m = 111000  # Approximately 111 km per degree
        dlat = (lat2 - lat1) * deg_to_m
        dlon = (lon2 - lon1) * deg_to_m * np.cos(np.radians((lat1 + lat2) / 2))
        
        return np.sqrt(dlat ** 2 + dlon ** 2)
    
    else:
        raise ValueError(f"Unknown distance method: {method}")


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    current_file = Path(__file__).resolve()
    
    # Go up the directory tree to find the project root
    for parent in current_file.parents:
        if (parent / 'setup.py').exists() or (parent / 'pyproject.toml').exists():
            return parent
    
    # Fallback: assume we're in src/utils and go up two levels
    return current_file.parents[2]


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} PB"


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay on each retry
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay:.1f}s...")
                    
                    import time
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator