from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum
import uuid
import random

# --- Enums and Types ---

class Side(str, Enum):
    WHITE = "white"
    BLACK = "black"

# --- Request Models ---

class OpeningStats(BaseModel):
    """Stats for a single opening from player's history"""
    opening_name: str
    opening_eco: str
    win_rate: float = Field(ge=0.0, le=1.0, description="Normalized win rate")
    num_games: int = Field(ge=0, description="Number of games played with this opening")
    # Add more fields as needed later
    
class PredictRequest(BaseModel):
    side: Side
    openings: List[OpeningStats] = Field(
        default_factory=list,
        description="List of openings with player's historical performance"
    )

# --- Response Models ---

class Recommendation(BaseModel):
    opening_name: str
    opening_eco: str
    predicted_score: float = Field(ge=0.0, le=1.0, description="Expected score (0-1)")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence")

class PredictResponse(BaseModel):
    request_id: str
    side: Side
    recommendations: List[Recommendation]
    model_loaded: bool = False
    model_version: str = "dummy"

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

# --- Dummy Data ---

DUMMY_OPENINGS_WHITE = [
    ("Italian Game", "C50"),
    ("Ruy Lopez", "C70"),
    ("Queen's Gambit", "D06"),
    ("English Opening", "A10"),
    ("King's Indian Attack", "A07"),
]

DUMMY_OPENINGS_BLACK = [
    ("Sicilian Defense", "B20"),
    ("French Defense", "C00"),
    ("Caro-Kann Defense", "B10"),
    ("Nimzo-Indian Defense", "E20"),
    ("King's Indian Defense", "E60"),
]

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
    Generate chess opening recommendations based on player history
    
    Currently returns dummy data. Will be replaced with actual model inference.
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Select dummy openings based on side
    dummy_pool = DUMMY_OPENINGS_WHITE if request.side == Side.WHITE else DUMMY_OPENINGS_BLACK
    
    # Generate dummy recommendations
    recommendations = []
    for opening_name, opening_eco in random.sample(dummy_pool, min(3, len(dummy_pool))):
        recommendations.append(
            Recommendation(
                opening_name=opening_name,
                opening_eco=opening_eco,
                predicted_score=round(random.uniform(0.45, 0.60), 3),
                confidence=round(random.uniform(0.70, 0.95), 3)
            )
        )
    
    # Sort by predicted score (descending)
    recommendations.sort(key=lambda x: x.predicted_score, reverse=True)
    
    return PredictResponse(
        request_id=request_id,
        side=request.side,
        recommendations=recommendations,
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