"""
Test utilities for the social infrastructure prediction system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from utils.helpers import (
    ensure_directory, load_config, save_config, validate_dataframe,
    split_train_test, calculate_distance, timing_context
)
from utils.exceptions import DataLoadingError, DataValidationError


class TestHelpers:
    """Test cases for helper utilities."""
    
    def test_ensure_directory(self, tmp_path):
        """Test directory creation."""
        test_dir = tmp_path / "test_dir" / "subdir"
        result = ensure_directory(test_dir)
        
        assert result.exists()
        assert result.is_dir()
        assert result == test_dir
    
    def test_validate_dataframe_valid(self):
        """Test DataFrame validation with valid data."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        })
        
        # Should not raise exception
        result = validate_dataframe(df, required_columns=['col1', 'col2'], min_rows=2)
        assert result is True
    
    def test_validate_dataframe_missing_columns(self):
        """Test DataFrame validation with missing columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df, required_columns=['col1', 'col2', 'col3'])
    
    def test_validate_dataframe_insufficient_rows(self):
        """Test DataFrame validation with insufficient rows."""
        df = pd.DataFrame({
            'col1': [1],
            'col2': ['a']
        })
        
        with pytest.raises(ValueError, match="must have at least"):
            validate_dataframe(df, min_rows=5)
    
    def test_validate_dataframe_not_dataframe(self):
        """Test DataFrame validation with non-DataFrame input."""
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            validate_dataframe([1, 2, 3])
    
    def test_split_train_test_with_target(self):
        """Test train-test split with target variable."""
        X = pd.DataFrame(np.random.randn(100, 4))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
    
    def test_split_train_test_without_target(self):
        """Test train-test split without target variable."""
        X = pd.DataFrame(np.random.randn(100, 4))
        
        X_train, X_test = split_train_test(X, test_size=0.3, random_state=42)
        
        assert len(X_train) == 70
        assert len(X_test) == 30
        assert isinstance(X_train, pd.DataFrame)
    
    def test_calculate_distance_haversine(self):
        """Test distance calculation using haversine formula."""
        # Distance between New York and Los Angeles (approximate)
        lat1, lon1 = 40.7128, -74.0060  # New York
        lat2, lon2 = 34.0522, -118.2437  # Los Angeles
        
        distance = calculate_distance(lat1, lon1, lat2, lon2, method='haversine')
        
        # Should be approximately 3944 km
        assert 3900000 < distance < 4000000  # in meters
    
    def test_calculate_distance_euclidean(self):
        """Test distance calculation using euclidean method."""
        lat1, lon1 = 40.0, -74.0
        lat2, lon2 = 40.01, -74.01
        
        distance = calculate_distance(lat1, lon1, lat2, lon2, method='euclidean')
        
        # Should be relatively small distance
        assert 0 < distance < 2000  # in meters
    
    def test_calculate_distance_invalid_method(self):
        """Test distance calculation with invalid method."""
        with pytest.raises(ValueError, match="Unknown distance method"):
            calculate_distance(40.0, -74.0, 41.0, -75.0, method='invalid')
    
    def test_timing_context(self):
        """Test timing context manager."""
        import time
        
        with timing_context() as timer:
            time.sleep(0.01)  # Sleep for 10ms
        
        assert hasattr(timer, 'elapsed')
        assert timer.elapsed >= 0.01  # Should be at least 10ms
        assert timer.start_time is not None
        assert timer.end_time is not None
    
    def test_load_config_yaml(self, tmp_path):
        """Test loading YAML configuration."""
        config_data = {
            'database': {'host': 'localhost', 'port': 5432},
            'api': {'timeout': 30}
        }
        
        config_file = tmp_path / "test_config.yaml"
        
        # Create test YAML file
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded_config = load_config(config_file)
        assert loaded_config == config_data
    
    def test_load_config_json(self, tmp_path):
        """Test loading JSON configuration."""
        config_data = {
            'database': {'host': 'localhost', 'port': 5432},
            'api': {'timeout': 30}
        }
        
        config_file = tmp_path / "test_config.json"
        
        # Create test JSON file
        import json
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loaded_config = load_config(config_file)
        assert loaded_config == config_data
    
    def test_load_config_nonexistent_file(self):
        """Test loading config from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.yaml")
    
    def test_load_config_unsupported_format(self, tmp_path):
        """Test loading config from unsupported format."""
        config_file = tmp_path / "test_config.txt"
        config_file.write_text("some text")
        
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            load_config(config_file)
    
    def test_save_config_yaml(self, tmp_path):
        """Test saving YAML configuration."""
        config_data = {
            'database': {'host': 'localhost', 'port': 5432},
            'api': {'timeout': 30}
        }
        
        config_file = tmp_path / "test_config.yaml"
        save_config(config_data, config_file)
        
        assert config_file.exists()
        
        # Load and verify
        loaded_config = load_config(config_file)
        assert loaded_config == config_data
    
    def test_save_config_json(self, tmp_path):
        """Test saving JSON configuration."""
        config_data = {
            'database': {'host': 'localhost', 'port': 5432},
            'api': {'timeout': 30}
        }
        
        config_file = tmp_path / "test_config.json"
        save_config(config_data, config_file)
        
        assert config_file.exists()
        
        # Load and verify
        loaded_config = load_config(config_file)
        assert loaded_config == config_data


# Fixtures for common test data
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': range(100),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def sample_bridge_data():
    """Create sample bridge data for testing."""
    return pd.DataFrame({
        'bridge_id': ['B001', 'B002', 'B003'],
        'name': ['Bridge A', 'Bridge B', 'Bridge C'],
        'latitude': [40.7128, 40.7580, 40.6892],
        'longitude': [-74.0060, -73.9855, -74.0445],
        'construction_year': [1985, 1978, 1992],
        'condition_score': [7.5, 6.2, 8.1],
        'maintenance_needed': [True, True, False]
    })


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([0.7, 0.3, 0.9])
    model.fit.return_value = model
    model.score.return_value = 0.85
    return model