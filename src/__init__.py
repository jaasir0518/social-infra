"""
Social Infrastructure Prediction Package

A comprehensive machine learning system for predicting infrastructure 
maintenance needs and investment priorities.
"""

__version__ = "0.1.0"
__author__ = "Social Infrastructure Team"
__email__ = "team@socialinfra.ai"

from .config.settings import Settings
from .utils.helpers import setup_logging

# Initialize logging
setup_logging()

__all__ = ["Settings", "setup_logging"]