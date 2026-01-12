from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum
import uuid
import pandas as pd
from pathlib import Path

from utils.inference.pipeline import PipelineConfig, run_foldin_pipeline



# --- Enums and Types ---

class Side(str, Enum):
    WHITE = "white"
    BLACK = "black"

# --- Request Models ---

class OpeningStatsRequest(BaseModel):
    """Stats for a single opening from player's history"""
    opening_name: str = Field(description="Full opening name")
    opening_id: int = Field(description="Training opening ID")
    eco: str = Field(description="ECO code (e.g., C50)")
    num_games: int = Field(ge=0, description="Games played with this opening")
    num_wins: int = Field(ge=0, description="Win count")
    num_draws: int = Field(ge=0, description="Draw count")
    num_losses: int = Field(ge=0, description="Loss count")

class PredictRequest(BaseModel):
    """Prediction request matching HFInterfacePayload from TS"""
    name: str = Field(description="Lichess username")
    rating: int = Field(ge=0, description="Player rating")
    side: Side = Field(description="White or Black")
    opening_stats: List[OpeningStatsRequest] = Field(
        default_factory=list,
        description="List of openings with player's historical performance"
    )

# --- Response Models ---

class Recommendation(BaseModel):
    """Opening recommendation from the model"""
    opening_name: str = Field(description="Full opening name")
    eco: str = Field(description="ECO code")
    predicted_score: float = Field(ge=0.0, le=1.0, description="Expected score (0-1)")

class RecommendationStats(BaseModel):
    """Statistics about the recommendations"""
    num_openings_total: int = Field(description="Total openings in training set")
    num_openings_played: int = Field(description="Openings player has played")
    num_openings_unplayed: int = Field(description="Openings player hasn't played")
    predicted_min: float = Field(description="Min predicted score across all openings")
    predicted_max: float = Field(description="Max predicted score across all openings")
    predicted_mean: float = Field(description="Mean predicted score across all openings")

class PredictResponse(BaseModel):
    """Recommendation response from the API"""
    request_id: str = Field(description="Unique request identifier")
    side: Side = Field(description="White or Black")
    recommendations: List[Recommendation] = Field(description="Top recommended openings")
    stats: RecommendationStats = Field(description="Statistics about predictions")
    model_loaded: bool = Field(description="Whether model was successfully loaded")
    model_version: str = Field(description="Model version/training date")

# --- FastAPI App ---

app = FastAPI(
    title="Chess Opening Recommender API",
    description="ML-powered chess opening recommendations for White and Black",
    version="0.1.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility Functions ---

def convert_predict_request_to_player_data(request: PredictRequest):
    """
    Convert FastAPI PredictRequest to PlayerData format expected by inference pipeline.
    
    Args:
        request: PredictRequest from API
        
    Returns:
        PlayerData object ready for pipeline processing
        
    Raises:
        ValueError: If opening_stats is empty or DataFrame validation fails
    """
    from utils.types.inference_pipeline_types import PlayerData
    
    # Convert opening_stats list to DataFrame
    opening_stats_list = []
    for opening in request.opening_stats:
        opening_stats_list.append({
            "opening_id": opening.opening_id,
            "eco": opening.eco,
            "opening_name": opening.opening_name,
            "num_games": opening.num_games,
            "num_wins": opening.num_wins,
            "num_draws": opening.num_draws,
            "num_losses": opening.num_losses,
        })
    
    opening_stats_df = pd.DataFrame(opening_stats_list)
    
    # Validate DataFrame has required columns
    required_columns = {
        "opening_id", "eco", "opening_name", 
        "num_games", "num_wins", "num_draws", "num_losses"
    }
    if not required_columns.issubset(opening_stats_df.columns):
        missing = required_columns - set(opening_stats_df.columns)
        raise ValueError(f"Missing required columns in opening_stats: {missing}")
    
    # Map side to color ('w' or 'b')
    color = 'w' if request.side == Side.WHITE else 'b'
    
    # Create PlayerData with dummy player_id for fold-in (new) users
    player_data = PlayerData(
        player_id=-1,  # Dummy ID for fold-in users (not in training set)
        name=request.name,
        rating=request.rating,
        color=color,
        opening_stats_df=opening_stats_df,
    )
    
    return player_data

# --- Endpoints ---

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Chess Opening Recommender API",
        "version": "0.1.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_white_loaded": False,
        "model_black_loaded": False,
        "model_version": "dummy"
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Generate chess opening recommendations based on player history."""
    player_data = convert_predict_request_to_player_data(request)
    request_id = str(uuid.uuid4())
    
    # Select artifacts based on side
    artifacts = _artifacts_white if request.side == Side.WHITE else _artifacts_black
    config = artifacts.config if artifacts else None
    
    if not artifacts or not config:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        result = run_foldin_pipeline(
            player_data=player_data,
            config=config,
            artifacts=artifacts
        )
        
        return PredictResponse(
            request_id=request_id,
            side=request.side,
            recommendations=[
                Recommendation(
                    opening_name=rec["opening_name"],
                    eco=rec["eco"],
                    predicted_score=rec["predicted_score"]
                )
                for rec in result["recommendations"]
            ],
            stats=RecommendationStats(
                num_openings_total=result["stats"]["num_openings_total"],
                num_openings_played=result["stats"]["num_openings_played"],
                num_openings_unplayed=result["stats"]["num_openings_unplayed"],
                predicted_min=result["stats"]["predicted_min"],
                predicted_max=result["stats"]["predicted_max"],
                predicted_mean=result["stats"]["predicted_mean"],
            ),
            model_loaded=True,
            model_version="production"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# --- Model Loading (Placeholder) ---

@app.on_event("startup")
# TODO maybe we just start up white or black later on, 
#   depending on which is needed?
async def load_models():
    """Load PyTorch models on startup"""
    global _artifacts_white, _artifacts_black
    from utils.inference.pipeline import PipelineArtifacts
    
    print("Starting Chess Opening Recommender API")
    artifacts_dir = Path(__file__).parent / "artifacts"
    
    try:
        _artifacts_white = PipelineArtifacts(PipelineConfig(
            model_artifacts_dir=artifacts_dir / "white",
            color="w",
        ))
        _artifacts_black = PipelineArtifacts(PipelineConfig(
            model_artifacts_dir=artifacts_dir / "black",
            color="b",
        ))
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("Shutting down Chess Opening Recommender API")