"""
Data API routes for social infrastructure prediction system.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class DataSummary(BaseModel):
    """Data summary response model."""
    infrastructure_type: str
    total_records: int
    last_updated: str
    data_quality_score: float
    

class InfrastructureRecord(BaseModel):
    """Infrastructure record model."""
    id: str
    name: str
    location: Dict[str, float]
    condition_score: float
    last_inspection: str
    

# Create router
data_router = APIRouter()


@data_router.get("/summary", response_model=List[DataSummary])
async def get_data_summary() -> List[DataSummary]:
    """
    Get summary of all infrastructure data.
    
    Returns:
        List of data summaries for each infrastructure type
    """
    try:
        summaries = [
            DataSummary(
                infrastructure_type="bridge",
                total_records=1250,
                last_updated="2023-09-28T08:00:00Z",
                data_quality_score=0.92
            ),
            DataSummary(
                infrastructure_type="housing",
                total_records=8500,
                last_updated="2023-09-28T07:30:00Z",
                data_quality_score=0.88
            ),
            DataSummary(
                infrastructure_type="road",
                total_records=3200,
                last_updated="2023-09-28T09:15:00Z",
                data_quality_score=0.85
            ),
            DataSummary(
                infrastructure_type="utility",
                total_records=5600,
                last_updated="2023-09-28T06:45:00Z",
                data_quality_score=0.90
            )
        ]
        
        return summaries
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get data summary: {str(e)}")


@data_router.get("/infrastructure/{infrastructure_type}", response_model=List[InfrastructureRecord])
async def get_infrastructure_data(
    infrastructure_type: str,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    condition_min: Optional[float] = Query(default=None, ge=0, le=10)
) -> List[InfrastructureRecord]:
    """
    Get infrastructure data by type.
    
    Args:
        infrastructure_type: Type of infrastructure
        limit: Maximum number of records to return
        offset: Number of records to skip
        condition_min: Minimum condition score filter
        
    Returns:
        List of infrastructure records
    """
    try:
        # Mock data - replace with actual database queries
        if infrastructure_type == "bridge":
            records = [
                InfrastructureRecord(
                    id="B001",
                    name="Main Street Bridge",
                    location={"lat": 40.7128, "lon": -74.0060},
                    condition_score=7.5,
                    last_inspection="2023-06-15"
                ),
                InfrastructureRecord(
                    id="B002",
                    name="River Crossing Bridge",
                    location={"lat": 40.7580, "lon": -73.9855},
                    condition_score=6.2,
                    last_inspection="2023-05-22"
                )
            ]
        elif infrastructure_type == "housing":
            records = [
                InfrastructureRecord(
                    id="H001",
                    name="Oak Street Complex",
                    location={"lat": 40.7200, "lon": -74.0100},
                    condition_score=8.2,
                    last_inspection="2023-07-10"
                )
            ]
        else:
            records = []
        
        # Apply filters
        if condition_min is not None:
            records = [r for r in records if r.condition_score >= condition_min]
        
        # Apply pagination
        end_idx = offset + limit
        return records[offset:end_idx]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get infrastructure data: {str(e)}")


@data_router.get("/statistics/{infrastructure_type}")
async def get_infrastructure_statistics(
    infrastructure_type: str
) -> Dict[str, Any]:
    """
    Get statistical information for infrastructure type.
    
    Args:
        infrastructure_type: Type of infrastructure
        
    Returns:
        Statistical summary
    """
    try:
        # Mock statistics - replace with actual calculations
        stats = {
            "total_count": 1250 if infrastructure_type == "bridge" else 5000,
            "average_condition": 7.2,
            "condition_distribution": {
                "excellent": 25,
                "good": 40,
                "fair": 25,
                "poor": 10
            },
            "maintenance_needed": 315,
            "average_age": 28.5,
            "newest": 5,
            "oldest": 65
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@data_router.post("/validate")
async def validate_data(
    infrastructure_type: str,
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate infrastructure data.
    
    Args:
        infrastructure_type: Type of infrastructure
        data: Data to validate
        
    Returns:
        Validation results
    """
    try:
        # Mock validation - replace with actual validation logic
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "data_quality_score": 0.95
        }
        
        # Basic validation checks
        if "condition_score" in data:
            score = data["condition_score"]
            if not (0 <= score <= 10):
                validation_results["valid"] = False
                validation_results["errors"].append("Condition score must be between 0 and 10")
        
        if "location" in data:
            location = data["location"]
            if "lat" not in location or "lon" not in location:
                validation_results["warnings"].append("Location missing latitude or longitude")
        
        return validation_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")