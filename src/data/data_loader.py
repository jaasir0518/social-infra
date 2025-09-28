"""
Data loading utilities for social infrastructure prediction system.
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import sqlite3
import psycopg2
from sqlalchemy import create_engine

from utils.exceptions import DataLoadingError, DataValidationError
from utils.constants import SUPPORTED_DATA_FORMATS
from config.logging_config import LoggingMixin


class DataLoader(LoggingMixin):
    """
    Handles loading data from various sources and formats.
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize DataLoader.
        
        Args:
            base_path: Base directory for data files
        """
        self.base_path = Path(base_path) if base_path else Path("data")
        self.supported_formats = SUPPORTED_DATA_FORMATS
        
    def load_csv(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv()
            
        Returns:
            Loaded DataFrame
        """
        try:
            full_path = self._resolve_path(file_path)
            self.logger.info(f"Loading CSV file: {full_path}")
            
            # Default parameters for CSV loading
            default_params = {
                'encoding': 'utf-8',
                'low_memory': False,
                'parse_dates': True,
                'infer_datetime_format': True
            }
            default_params.update(kwargs)
            
            df = pd.read_csv(full_path, **default_params)
            self.logger.info(f"Successfully loaded {len(df)} rows from {full_path}")
            
            return df
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load CSV file {file_path}: {e}")
    
    def load_excel(
        self,
        file_path: Union[str, Path],
        sheet_name: Optional[Union[str, int]] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load data from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index to load
            **kwargs: Additional arguments for pd.read_excel()
            
        Returns:
            Loaded DataFrame or dictionary of DataFrames
        """
        try:
            full_path = self._resolve_path(file_path)
            self.logger.info(f"Loading Excel file: {full_path}")
            
            df = pd.read_excel(full_path, sheet_name=sheet_name, **kwargs)
            
            if isinstance(df, dict):
                self.logger.info(f"Successfully loaded {len(df)} sheets from {full_path}")
            else:
                self.logger.info(f"Successfully loaded {len(df)} rows from {full_path}")
            
            return df
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load Excel file {file_path}: {e}")
    
    def load_json(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments for pd.read_json()
            
        Returns:
            Loaded DataFrame
        """
        try:
            full_path = self._resolve_path(file_path)
            self.logger.info(f"Loading JSON file: {full_path}")
            
            df = pd.read_json(full_path, **kwargs)
            self.logger.info(f"Successfully loaded {len(df)} rows from {full_path}")
            
            return df
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load JSON file {file_path}: {e}")
    
    def load_geospatial(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> gpd.GeoDataFrame:
        """
        Load geospatial data (GeoJSON, Shapefile, etc.).
        
        Args:
            file_path: Path to geospatial file
            **kwargs: Additional arguments for gpd.read_file()
            
        Returns:
            Loaded GeoDataFrame
        """
        try:
            full_path = self._resolve_path(file_path)
            self.logger.info(f"Loading geospatial file: {full_path}")
            
            gdf = gpd.read_file(full_path, **kwargs)
            self.logger.info(f"Successfully loaded {len(gdf)} features from {full_path}")
            
            return gdf
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load geospatial file {file_path}: {e}")
    
    def load_parquet(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments for pd.read_parquet()
            
        Returns:
            Loaded DataFrame
        """
        try:
            full_path = self._resolve_path(file_path)
            self.logger.info(f"Loading Parquet file: {full_path}")
            
            df = pd.read_parquet(full_path, **kwargs)
            self.logger.info(f"Successfully loaded {len(df)} rows from {full_path}")
            
            return df
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load Parquet file {file_path}: {e}")
    
    def load_from_database(
        self,
        query: str,
        connection_string: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from database using SQL query.
        
        Args:
            query: SQL query to execute
            connection_string: Database connection string
            **kwargs: Additional arguments for pd.read_sql()
            
        Returns:
            Loaded DataFrame
        """
        try:
            self.logger.info(f"Executing database query: {query[:100]}...")
            
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine, **kwargs)
            
            self.logger.info(f"Successfully loaded {len(df)} rows from database")
            
            return df
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load data from database: {e}")
    
    def load_multiple_files(
        self,
        file_pattern: str,
        loader_func: str = "load_csv",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load and concatenate multiple files matching a pattern.
        
        Args:
            file_pattern: Glob pattern for files to load
            loader_func: Name of loader function to use
            **kwargs: Additional arguments for the loader function
            
        Returns:
            Concatenated DataFrame
        """
        try:
            base_path = self.base_path / file_pattern if not os.path.isabs(file_pattern) else Path(file_pattern)
            files = list(base_path.parent.glob(base_path.name))
            
            if not files:
                raise DataLoadingError(f"No files found matching pattern: {file_pattern}")
            
            self.logger.info(f"Loading {len(files)} files matching pattern: {file_pattern}")
            
            loader = getattr(self, loader_func)
            dataframes = []
            
            for file_path in files:
                try:
                    df = loader(file_path, **kwargs)
                    df['source_file'] = file_path.name
                    dataframes.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
            
            if not dataframes:
                raise DataLoadingError("No files could be loaded successfully")
            
            combined_df = pd.concat(dataframes, ignore_index=True)
            self.logger.info(f"Successfully combined {len(combined_df)} rows from {len(dataframes)} files")
            
            return combined_df
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load multiple files: {e}")
    
    def load_infrastructure_data(
        self,
        infrastructure_type: str,
        data_sources: Optional[Dict[str, str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all data for a specific infrastructure type.
        
        Args:
            infrastructure_type: Type of infrastructure (bridge, housing, road, utility)
            data_sources: Dictionary mapping data types to file paths
            
        Returns:
            Dictionary of loaded DataFrames
        """
        try:
            self.logger.info(f"Loading data for {infrastructure_type} infrastructure")
            
            if data_sources is None:
                data_sources = self._get_default_data_sources(infrastructure_type)
            
            loaded_data = {}
            
            for data_type, file_path in data_sources.items():
                try:
                    # Determine file format and use appropriate loader
                    file_path = Path(file_path)
                    
                    if file_path.suffix.lower() in ['.csv']:
                        df = self.load_csv(file_path)
                    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                        df = self.load_excel(file_path)
                    elif file_path.suffix.lower() in ['.json']:
                        df = self.load_json(file_path)
                    elif file_path.suffix.lower() in ['.geojson', '.shp']:
                        df = self.load_geospatial(file_path)
                    elif file_path.suffix.lower() in ['.parquet']:
                        df = self.load_parquet(file_path)
                    else:
                        self.logger.warning(f"Unsupported file format: {file_path}")
                        continue
                    
                    loaded_data[data_type] = df
                    self.logger.info(f"Loaded {data_type} data: {len(df)} rows")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {data_type} data from {file_path}: {e}")
            
            if not loaded_data:
                raise DataLoadingError(f"No data could be loaded for {infrastructure_type}")
            
            self.logger.info(f"Successfully loaded {len(loaded_data)} datasets for {infrastructure_type}")
            
            return loaded_data
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load infrastructure data: {e}")
    
    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        """
        Resolve file path relative to base path.
        
        Args:
            file_path: Input file path
            
        Returns:
            Resolved absolute path
        """
        path = Path(file_path)
        
        if path.is_absolute():
            return path
        else:
            return self.base_path / path
    
    def _get_default_data_sources(self, infrastructure_type: str) -> Dict[str, str]:
        """
        Get default data sources for an infrastructure type.
        
        Args:
            infrastructure_type: Type of infrastructure
            
        Returns:
            Dictionary mapping data types to default file paths
        """
        base_path = f"data/raw/{infrastructure_type}"
        
        default_sources = {
            "bridge": {
                "inventory": f"{base_path}/bridge_inventory.csv",
                "conditions": f"{base_path}/bridge_conditions.csv",
                "coordinates": f"{base_path}/bridge_coordinates.geojson"
            },
            "housing": {
                "board_data": f"{base_path}/housing_board_data.csv",
                "property_records": f"{base_path}/property_records.csv",
                "locations": f"{base_path}/housing_locations.geojson"
            },
            "road": {
                "network": f"{base_path}/road_network.csv",
                "traffic": f"{base_path}/traffic_data.csv",
                "conditions": f"{base_path}/road_conditions.csv"
            },
            "utility": {
                "water": f"{base_path}/water_supply.csv",
                "electricity": f"{base_path}/electricity_grid.csv",
                "sewage": f"{base_path}/sewage_systems.csv"
            }
        }
        
        return default_sources.get(infrastructure_type, {})