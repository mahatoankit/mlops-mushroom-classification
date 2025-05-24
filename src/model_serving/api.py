"""
API server for serving mushroom classification predictions.
This module provides a FastAPI application for serving predictions from the trained model.
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Add the project directory to the Python path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model_serving.model_loader import MushroomModel
from src.ab_testing import get_model_for_request, record_prediction_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    config_path = "config/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded configuration from {config_path}")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    config = {"paths": {"models": "models"}}

# Define the application
app = FastAPI(
    title="Mushroom Classification API",
    description="API for predicting mushroom edibility based on various features",
    version="1.0.0",
)

# We'll load models dynamically based on A/B testing configuration
# For now, we'll just keep a cache of models
model_cache = {}


# Input schema definition
class MushroomFeatures(BaseModel):
    """Mushroom features for prediction."""

    # Categorical features
    cap_shape: Optional[str] = Field(None, description="Shape of the cap")
    cap_color: Optional[str] = Field(None, description="Color of the cap")
    cap_surface: Optional[str] = Field(None, description="Surface texture of the cap")
    gill_color: Optional[str] = Field(None, description="Color of the gills")
    gill_attachment: Optional[str] = Field(None, description="Gill attachment type")
    stem_color: Optional[str] = Field(None, description="Color of the stem")
    ring_type: Optional[str] = Field(None, description="Type of ring on the stem")
    habitat: Optional[str] = Field(
        None, description="Habitat where the mushroom is found"
    )

    # Numeric features
    cap_diameter: Optional[float] = Field(None, description="Diameter of the cap in cm")
    stem_height: Optional[float] = Field(None, description="Height of the stem in cm")
    stem_width: Optional[float] = Field(None, description="Width of the stem in cm")

    # Binary features
    does_bruise_or_bleed: Optional[bool] = Field(
        None, description="Whether the mushroom bruises or bleeds"
    )
    has_ring: Optional[bool] = Field(
        None, description="Whether the mushroom has a ring"
    )

    class Config:
        schema_extra = {
            "example": {
                "cap_shape": "convex",
                "cap_color": "brown",
                "cap_surface": "smooth",
                "gill_color": "white",
                "gill_attachment": "free",
                "stem_color": "white",
                "ring_type": "pendant",
                "habitat": "woods",
                "cap_diameter": 5.7,
                "stem_height": 8.2,
                "stem_width": 1.5,
                "does_bruise_or_bleed": True,
                "has_ring": True,
            }
        }


# Output schema definition
class PredictionResult(BaseModel):
    """Result of mushroom classification prediction."""

    prediction: List[int] = Field(
        ..., description="Raw numerical prediction (0=Poisonous, 1=Edible)"
    )
    prediction_label: List[str] = Field(
        ..., description="Human-readable prediction labels"
    )
    probability: Optional[List[float]] = Field(
        None, description="Probability of the positive class (Edible)"
    )
    class_names: List[str] = Field(
        ..., description="Names of the classes (Poisonous, Edible)"
    )


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {"message": "Mushroom Classification API is running"}


@app.post("/predict", response_model=PredictionResult)
async def predict(
    features: MushroomFeatures,
    ab_test: Optional[str] = Query(None, description="Name of A/B test to use"),
    ground_truth: Optional[int] = Query(
        None, description="Ground truth for model evaluation (0=Poisonous, 1=Edible)"
    ),
):
    """
    Make a prediction based on the provided mushroom features.
    Optional A/B testing parameter allows routing to a specific test.
    """
    try:
        logger.info("Received prediction request")

        # Get appropriate model based on A/B testing configuration
        model_path, test_id = get_model_for_request(ab_test)

        # Load model from cache or create new
        if model_path not in model_cache:
            model_type = "xgboost"  # Default, could be inferred from filename
            model_cache[model_path] = MushroomModel(model_path, model_type=model_type)
            logger.info(f"Loaded model from {model_path}")

        # Get model from cache
        current_model = model_cache[model_path]

        # Convert Pydantic model to dict
        feature_dict = features.model_dump()

        # Make prediction
        result = current_model.predict(feature_dict)

        # Add model info to result
        result["model_type"] = current_model.model_type
        result["model_path"] = model_path

        # Record A/B test result if applicable
        if test_id:
            model_key = "A" if ab_test and "model_a" in model_path else "B"
            record_prediction_result(
                test_id=test_id,
                model=model_key,
                prediction=result["prediction"][0],
                ground_truth=ground_truth,
                metadata={
                    "features": feature_dict,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            result["ab_test"] = {"test_id": test_id, "model": model_key}

        # Store prediction in database (if enabled)
        try:
            from src.model_serving.database import DatabaseClient

            # Initialize database client
            db_client = DatabaseClient("oltp")

            # Store prediction
            db_client.store_prediction(feature_dict, result)

            # Close connection
            db_client.close()

            logger.info("Stored prediction in database")
        except Exception as db_error:
            # Log error but don't fail the prediction request
            logger.error(f"Error storing prediction in database: {db_error}")

        logger.info(f"Successful prediction: {result['prediction_label']}")
        return result

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error making prediction: {str(e)}"
        )


@app.get("/ab-tests")
async def list_ab_tests():
    """List all active A/B tests."""
    try:
        from src.ab_testing import ABTestRegistry

        registry = ABTestRegistry()
        active_tests = registry.list_active_tests()
        all_tests = registry.list_all_tests()

        return {"active_tests": active_tests, "all_tests": all_tests}
    except Exception as e:
        logger.error(f"Error listing A/B tests: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error listing A/B tests: {str(e)}"
        )


@app.post("/ab-tests")
async def create_ab_test(
    name: str = Query(..., description="Name for the A/B test"),
    model_a: str = Query(..., description="Path or name of model A (control)"),
    model_b: str = Query(..., description="Path or name of model B (variant)"),
    traffic_split: float = Query(
        0.5, description="Percentage of traffic to route to model B (0-1)"
    ),
):
    """Create a new A/B test."""
    try:
        from src.ab_testing import create_ab_test

        test_id = create_ab_test(name, model_a, model_b, traffic_split)
        return {"test_id": test_id, "status": "created"}
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error creating A/B test: {str(e)}"
        )


@app.get("/ab-tests/{test_id}")
async def get_ab_test(test_id: str):
    """Get information about a specific A/B test."""
    try:
        from src.ab_testing import ABTestRegistry

        registry = ABTestRegistry()
        test = registry.get_test(test_id)

        if test is None:
            # Check if it's a completed test
            all_tests = registry.list_all_tests()
            completed_test = next((t for t in all_tests if t["id"] == test_id), None)

            if completed_test:
                return {"status": "completed", "test": completed_test}

            raise HTTPException(
                status_code=404, detail=f"A/B test with ID {test_id} not found"
            )

        return {
            "id": test.id,
            "name": test.name,
            "model_a": test.model_a,
            "model_b": test.model_b,
            "traffic_split": test.traffic_split,
            "status": test.status,
            "start_time": test.start_time.isoformat(),
            "sample_sizes": {"a": len(test.results_a), "b": len(test.results_b)},
            "metrics": {"a": test.metrics_a, "b": test.metrics_b},
            "comparison": test.comparison,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting A/B test: {str(e)}")


@app.post("/ab-tests/{test_id}/conclude")
async def conclude_ab_test(
    test_id: str,
    winner: Optional[str] = Query(
        None, description="Force winner selection ('A' or 'B')"
    ),
):
    """Conclude an A/B test and optionally select a winner."""
    try:
        from src.ab_testing import ABTestRegistry

        registry = ABTestRegistry()
        conclusion = registry.conclude_test(test_id, winner)

        if "status" in conclusion and conclusion["status"] == "error":
            raise HTTPException(status_code=404, detail=conclusion["message"])

        return {"status": "concluded", "conclusion": conclusion}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error concluding A/B test: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error concluding A/B test: {str(e)}"
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    # Ensure default model is loaded
    model_path, _ = get_model_for_request()

    if model_path not in model_cache:
        model_type = "xgboost"  # Default
        model_cache[model_path] = MushroomModel(model_path, model_type=model_type)

    return {
        "status": "ok",
        "models_loaded": len(model_cache),
        "default_model": model_path,
    }


# Run the API with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
