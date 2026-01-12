from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum
import uuid
import pandas as pd
from pathlib import Path

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
    """
    Generate chess opening recommendations based on player history.
    
    Currently returns dummy data. Will be replaced with actual model inference.
    """
    # Convert request to PlayerData format (for validation)
    player_data = convert_predict_request_to_player_data(request)
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    print(f"Processing prediction for player: {player_data.name} "
          f"(Rating: {player_data.rating}, Color: {player_data.color})")
    
    # Dummy openings for each side
    dummy_openings_white = [
        ("Italian Game", "C50", 0.52),
        ("Ruy Lopez", "C70", 0.50),
        ("Queen's Gambit", "D06", 0.51),
    ]
    
    dummy_openings_black = [
        ("Sicilian Defense", "B20", 0.53),
        ("French Defense", "C00", 0.51),
        ("Caro-Kann Defense", "B10", 0.50),
    ]
    
    # Select dummy openings based on side
    dummy_pool = dummy_openings_white if request.side == Side.WHITE else dummy_openings_black
    
    # Generate dummy recommendations
    recommendations = []
    for opening_name, eco, score in dummy_pool:
        recommendations.append(
            Recommendation(
                opening_name=opening_name,
                eco=eco,
                predicted_score=score,
            )
        )
    
    # Sort by predicted score (descending)
    recommendations.sort(key=lambda x: x.predicted_score, reverse=True)
    
    return PredictResponse(
        request_id=request_id,
        side=request.side,
        recommendations=recommendations,
        stats=RecommendationStats(
            num_openings_total=600,
            num_openings_played=len(request.opening_stats),
            num_openings_unplayed=1700 - len(request.opening_stats),
            predicted_min=0.45,
            predicted_max=0.65,
            predicted_mean=0.52,
        ),
        model_loaded=False,
        model_version="dummy"
    )

# --- Model Loading (Placeholder) ---

@app.on_event("startup")
async def load_models():
    """
    Load PyTorch models on startup
    
    TODO: Implement actual model loading from .pt files
    """
    print("Starting Chess Opening Recommender API")
    print("Running with dummy predictions (models not loaded)")
    # Future: Load white_model.pt and black_model.pt here

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("Shutting down Chess Opening Recommender API")