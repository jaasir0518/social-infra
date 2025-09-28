"""
Custom exceptions for the social infrastructure prediction system.
"""


class SocialInfraError(Exception):
    """Base exception class for social infrastructure prediction system."""
    pass


class ConfigurationError(SocialInfraError):
    """Raised when there's an issue with configuration."""
    pass


class DataError(SocialInfraError):
    """Base exception for data-related errors."""
    pass


class DataLoadingError(DataError):
    """Raised when data cannot be loaded."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataProcessingError(DataError):
    """Raised when data processing fails."""
    pass


class DataQualityError(DataError):
    """Raised when data quality is below acceptable standards."""
    pass


class GeospatialError(DataError):
    """Raised when geospatial data processing fails."""
    pass


class ModelError(SocialInfraError):
    """Base exception for model-related errors."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class ModelLoadingError(ModelError):
    """Raised when model cannot be loaded."""
    pass


class ModelSavingError(ModelError):
    """Raised when model cannot be saved."""
    pass


class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    pass


class FeatureEngineeringError(SocialInfraError):
    """Raised when feature engineering fails."""
    pass


class EvaluationError(SocialInfraError):
    """Raised when model evaluation fails."""
    pass


class APIError(SocialInfraError):
    """Base exception for API-related errors."""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    pass


class ValidationError(APIError):
    """Raised when API input validation fails."""
    pass


class ResourceNotFoundError(APIError):
    """Raised when requested resource is not found."""
    pass


class DatabaseError(SocialInfraError):
    """Base exception for database-related errors."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class QueryError(DatabaseError):
    """Raised when database query fails."""
    pass


class DeploymentError(SocialInfraError):
    """Raised when deployment fails."""
    pass


class MonitoringError(SocialInfraError):
    """Raised when monitoring setup fails."""
    pass