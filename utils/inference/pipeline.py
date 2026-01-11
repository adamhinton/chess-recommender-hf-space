"""
Fold-in inference data processing pipeline.

This module transforms raw player data (from DB or Lichess API) into
model-ready input tensors for making opening recommendations.

The pipeline is designed to be:
- Reusable (local testing and production)
- Color-agnostic (handles both White and Black openings)
- Type-safe (using dataclasses and type hints)
- Production-ready (proper error handling and logging)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import pandas as pd
import numpy as np

# import torch
# import torch.nn as nn

from utils.types.inference_pipeline_types import PlayerData, ModelInput

@dataclass
class PipelineConfig:
    """Configuration for the fold-in data processing pipeline."""

    # Model artifacts directory
    model_artifacts_dir: Path

    # Color being processed ('w' for White, 'b' for Black)
    color: str

    # Filtering thresholds
    min_games_threshold: int = 3

    # Bayesian shrinkage strength (must match training)
    k_shrinkage: int = 50

    # Verbosity level (0=silent, 1=progress, 2=detailed)
    verbose: int = 1

    # Inference configuration
    top_n_recommendations: int = 50
    inference_batch_size: int = 512
    device: str = "auto"  # "auto", "cpu", "cuda"

    def __post_init__(self):
        """Validate configuration."""
        if self.color not in ("w", "b"):
            raise ValueError(f"color must be 'w' or 'b', got '{self.color}'")

        if not self.model_artifacts_dir.exists():
            raise FileNotFoundError(
                f"Model artifacts directory not found: {self.model_artifacts_dir}"
            )

        if self.min_games_threshold < 1:
            raise ValueError(
                f"min_games_threshold must be >= 1, got {self.min_games_threshold}"
            )

        if self.k_shrinkage < 1:
            raise ValueError(f"k_shrinkage must be >= 1, got {self.k_shrinkage}")

        if self.top_n_recommendations < 1:
            raise ValueError(
                f"top_n_recommendations must be >= 1, got {self.top_n_recommendations}"
            )

        if self.inference_batch_size < 1:
            raise ValueError(
                f"inference_batch_size must be >= 1, got {self.inference_batch_size}"
            )

        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"device must be one of ['auto', 'cpu', 'cuda'], got '{self.device}'"
            )


def _resolve_torch_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# BIG TODO: HF has its own way of loading models; 
#   will need to make sure whatever we do here is compatible with that
class ChessOpeningRecommender(nn.Module):
    """Model architecture (copied from notebook 28/31 to ensure exact match).

    Notes:
        - The model uses buffers for all opening-level ECO features and player ratings.
        - Forward pass takes only (player_ids, opening_ids).
    """

    def __init__(
        self,
        num_players: int,
        num_openings: int,
        num_factors: int,
        player_ratings: torch.Tensor,
        opening_eco_letters: torch.Tensor,
        opening_eco_numbers: torch.Tensor,
        num_eco_letters: int,
        num_eco_numbers: int,
        eco_embed_dim: int,
    ):
        super().__init__()

        # Store side information as buffers (not trainable parameters)
        self.register_buffer("player_ratings", player_ratings)
        self.register_buffer("opening_eco_letters", opening_eco_letters)
        self.register_buffer("opening_eco_numbers", opening_eco_numbers)

        # Embeddings
        self.player_embedding = nn.Embedding(num_players, num_factors)
        self.opening_embedding = nn.Embedding(num_openings, num_factors)

        # ECO embeddings
        self.eco_letter_embedding = nn.Embedding(num_eco_letters, eco_embed_dim)
        self.eco_number_embedding = nn.Embedding(num_eco_numbers, eco_embed_dim)

        # Combine opening factors with ECO embeddings
        # Input dim: num_factors + 2*eco_embed_dim -> output dim: num_factors
        self.opening_combiner = nn.Linear(num_factors + 2 * eco_embed_dim, num_factors)

        # Bias terms
        self.player_biases = nn.Embedding(num_players, 1)
        self.opening_biases = nn.Embedding(num_openings, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, player_ids: torch.Tensor, opening_ids: torch.Tensor
    ) -> torch.Tensor:
        # Player representation
        player_embed = self.player_embedding(player_ids)  # [batch, num_factors]
        rating = self.player_ratings[player_ids].unsqueeze(1)  # [batch, 1]
        player_repr = player_embed + rating

        # Opening representation
        opening_embed = self.opening_embedding(opening_ids)  # [batch, num_factors]
        eco_letters = self.opening_eco_letters[opening_ids]
        eco_numbers = self.opening_eco_numbers[opening_ids]
        eco_letter_embed = self.eco_letter_embedding(eco_letters)
        eco_number_embed = self.eco_number_embedding(eco_numbers)

        opening_concat = torch.cat(
            [opening_embed, eco_letter_embed, eco_number_embed], dim=1
        )
        opening_repr = self.opening_combiner(opening_concat)

        # Bias
        player_bias = self.player_biases(player_ids).squeeze()
        opening_bias = self.opening_biases(opening_ids).squeeze()

        # Interaction + sigmoid
        interaction = (player_repr * opening_repr).sum(dim=1)
        prediction = interaction + player_bias + opening_bias + self.global_bias
        return torch.sigmoid(prediction)


class PipelineArtifacts:
    """Container for loaded model artifacts (loaded once, reused many times)."""

    def __init__(self, config: PipelineConfig):
        """Load all required artifacts from disk."""
        self.config = config

        # Load opening mappings
        opening_mappings_path = config.model_artifacts_dir / "opening_mappings.csv"
        self.opening_mappings_df = pd.read_csv(opening_mappings_path)
        self.valid_opening_db_ids = set(self.opening_mappings_df["db_id"].values)
        self.db_to_training_id = dict(
            zip(
                self.opening_mappings_df["db_id"],
                self.opening_mappings_df["training_id"],
            )
        )

        # Load rating normalization params
        rating_norm_path = config.model_artifacts_dir / "rating_normalization.json"
        with open(rating_norm_path, "r") as f:
            rating_norm = json.load(f)
        self.rating_mean = rating_norm["rating_mean"]
        self.rating_std = rating_norm["rating_std"]

        # Load opening stats for Bayesian shrinkage
        color_name = "white" if config.color == "w" else "black"
        opening_stats_path = (
            config.model_artifacts_dir / f"opening_stats_{color_name}.json"
        )
        with open(opening_stats_path, "r") as f:
            self.opening_stats = json.load(f)

        # Load ECO encodings
        eco_encodings_path = config.model_artifacts_dir / "eco_encodings.json"
        with open(eco_encodings_path, "r") as f:
            eco_encodings = json.load(f)
        self.eco_letter_map = eco_encodings["eco_letter_to_int"]
        self.eco_number_map = eco_encodings["eco_number_to_int"]

        # Load hyperparameters needed for inference
        hyperparams_path = config.model_artifacts_dir / "hyperparameters.json"
        with open(hyperparams_path, "r") as f:
            self.hyperparams = json.load(f)

        # These are for convenience/clarity
        self.num_players = int(self.hyperparams["num_players"])
        self.num_openings = int(self.hyperparams["num_openings"])
        self.num_factors = int(self.hyperparams["num_factors"])
        self.num_eco_letters = int(self.hyperparams["num_eco_letters"])
        self.num_eco_numbers = int(self.hyperparams["num_eco_numbers"])
        self.eco_embed_dim = int(self.hyperparams["eco_embed_dim"])

        # Lazy-loaded model (created on first inference)
        self._model: Optional[nn.Module] = None

        if config.verbose >= 1:
            print(f"✓ Loaded artifacts for {color_name} openings")
            print(f"  • {len(self.valid_opening_db_ids):,} valid openings")
            print(
                f"  • Rating norm: mean={self.rating_mean:.0f}, std={self.rating_std:.0f}"
            )
            print(f"  • Opening stats: {len(self.opening_stats):,} entries")
            print(
                "  • Hyperparams: "
                f"players={self.num_players:,}, openings={self.num_openings:,}, factors={self.num_factors}"
            )


def _numpy_to_torch(
    array: np.ndarray, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.tensor(array, device=device, dtype=dtype)


def load_inference_model(
    artifacts: PipelineArtifacts,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load and cache the trained model for inference."""
    if artifacts._model is not None:
        return artifacts._model

    config = artifacts.config
    device = device or _resolve_torch_device(config.device)

    if config.verbose >= 1:
        print("Loading inference model...")

    # Build opening-level side info from mappings DF
    # These are aligned on training_id 0..NUM_OPENINGS-1
    opening_df = artifacts.opening_mappings_df.sort_values("training_id")
    all_eco_letter_cats = _numpy_to_torch(
        opening_df["eco"]
        .str[0]
        .map(artifacts.eco_letter_map)
        .fillna(0)
        .astype(int)
        .to_numpy(),
        device=device,
        dtype=torch.long,
    )
    all_eco_number_cats = _numpy_to_torch(
        opening_df["eco"]
        .str[1:]
        .map(artifacts.eco_number_map)
        .fillna(0)
        .astype(int)
        .to_numpy(),
        device=device,
        dtype=torch.long,
    )

    # Player ratings buffer: for fold-in users we only need a placeholder.
    # Keep size = NUM_PLAYERS to match embedding tables.
    player_ratings = torch.zeros(
        artifacts.num_players, dtype=torch.float32, device=device
    )

    model = ChessOpeningRecommender(
        num_players=artifacts.num_players,
        num_openings=artifacts.num_openings,
        num_factors=artifacts.num_factors,
        player_ratings=player_ratings,
        opening_eco_letters=all_eco_letter_cats,
        opening_eco_numbers=all_eco_number_cats,
        num_eco_letters=artifacts.num_eco_letters,
        num_eco_numbers=artifacts.num_eco_numbers,
        eco_embed_dim=artifacts.eco_embed_dim,
    )

    model_path = config.model_artifacts_dir / "best_model.pt"
    state_dict = torch.load(model_path, map_location=device)
    # Remap old checkpoint keys to match current architecture
    key_remap = {
        "player_factors.weight": "player_embedding.weight",
        "opening_factors.weight": "opening_embedding.weight",
    }
    for old_key, new_key in key_remap.items():
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
    # Drop unexpected keys from old checkpoints (player_combiner was removed)
    for extra_key in list(state_dict.keys()):
        if extra_key.startswith("player_combiner"):
            del state_dict[extra_key]
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    artifacts._model = model

    if config.verbose >= 1:
        print(f"✓ Loaded model weights: {model_path.name}")
        print(f"  • Device: {device}")

    return model


@torch.no_grad()
def predict_all_openings(
    *,
    artifacts: PipelineArtifacts,
    model_input: ModelInput,
    model: Optional[nn.Module] = None,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Predict scores for every training opening (not just played ones)."""
    config = artifacts.config
    device = _resolve_torch_device(config.device)
    model = model or load_inference_model(artifacts, device=device)
    batch_size = batch_size or config.inference_batch_size

    num_openings = artifacts.num_openings
    all_opening_ids = torch.arange(num_openings, dtype=torch.long, device=device)

    # Fold-in users use player_id=0 (same convention as notebook 31)
    player_id = (
        0
        if model_input.training_player_id is None
        else int(model_input.training_player_id)
    )

    all_predictions: list[np.ndarray] = []
    for start in range(0, num_openings, batch_size):
        end = min(start + batch_size, num_openings)
        batch_len = end - start

        batch_opening_ids = all_opening_ids[start:end]
        batch_player_ids = torch.full(
            (batch_len,), player_id, dtype=torch.long, device=device
        )

        preds = model(player_ids=batch_player_ids, opening_ids=batch_opening_ids)
        preds = preds.detach().cpu()
        if preds.dim() > 1:
            preds = preds.squeeze()
        all_predictions.append(np.array(preds.tolist()))

    return np.concatenate(all_predictions)


def rank_recommendations(
    *,
    artifacts: PipelineArtifacts,
    predictions: np.ndarray,
    played_opening_ids: Optional[np.ndarray],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (played_df, unplayed_df) with predicted scores."""
    results_df = artifacts.opening_mappings_df.copy()
    results_df["predicted_score"] = predictions

    if played_opening_ids is not None:
        played_mask = results_df["training_id"].isin(played_opening_ids)
        played_df = results_df[played_mask].sort_values(
            "predicted_score", ascending=False
        )
        unplayed_df = results_df[~played_mask].sort_values(
            "predicted_score", ascending=False
        )
    else:
        played_df = pd.DataFrame()
        unplayed_df = results_df.sort_values("predicted_score", ascending=False)

    return played_df, unplayed_df.head(top_n)


def build_recommendations_output(
    *,
    artifacts: PipelineArtifacts,
    model_input: ModelInput,
    predictions: np.ndarray,
) -> dict:
    """Create a stable, JSON-serializable result payload."""
    config = artifacts.config

    played_df, unplayed_df = rank_recommendations(
        artifacts=artifacts,
        predictions=predictions,
        played_opening_ids=model_input.opening_ids,
        top_n=config.top_n_recommendations,
    )

    # Attach actual scores for played openings (if any) for quick sanity checks
    if len(played_df) > 0:
        played_df = played_df.copy().set_index("training_id")
        actual_scores_map = dict(zip(model_input.opening_ids, model_input.scores))
        played_df["actual_score"] = played_df.index.map(actual_scores_map)
        played_df = played_df.reset_index()

    recommendations = (
        unplayed_df[["eco", "name", "predicted_score"]]
        .rename(columns={"name": "opening_name"})
        .to_dict(orient="records")
    )

    return {
        "player": {
            "training_player_id": model_input.training_player_id,
            "rating_z": float(model_input.rating_z),
            "color": config.color,
        },
        "stats": {
            "num_openings_total": int(artifacts.num_openings),
            "num_openings_played": int(len(model_input)),
            "num_openings_unplayed": int(artifacts.num_openings - len(model_input)),
            "predicted_min": float(np.min(predictions)),
            "predicted_max": float(np.max(predictions)),
            "predicted_mean": float(np.mean(predictions)),
            "top_n": int(config.top_n_recommendations),
        },
        "recommendations": recommendations,
        # Keep these optional/diagnostic (can be large). Only include if verbose >= 2.
        "diagnostics": {
            "played_openings": (
                played_df.to_dict(orient="records") if config.verbose >= 2 else None
            )
        },
    }


def filter_valid_openings(
    player_data: PlayerData,
    artifacts: PipelineArtifacts,
) -> PlayerData:
    """
    Filter player's openings to only include those in training set
    and meeting minimum games threshold.
    """
    config = artifacts.config
    df = player_data.opening_stats_df

    if config.verbose >= 2:
        print(f"\n{'='*80}")
        print("STEP: Filtering Valid Openings")
        print(f"{'='*80}")
        print(f"  Original: {len(df)} openings, {df['num_games'].sum():,} games")

    # Filter: in training set
    in_training = df["opening_id"].isin(artifacts.valid_opening_db_ids)
    df = df[in_training].copy()

    # Filter: minimum games
    meets_threshold = df["num_games"] >= config.min_games_threshold
    df = df[meets_threshold].copy()

    if len(df) == 0:
        raise ValueError(
            f"No valid openings remaining for player {player_data.name}! "
            f"All openings were either not in training set or had < {config.min_games_threshold} games."
        )

    if config.verbose >= 2:
        print(f"  Filtered: {len(df)} openings, {df['num_games'].sum():,} games")

    return PlayerData(
        player_id=player_data.player_id,
        name=player_data.name,
        rating=player_data.rating,
        color=player_data.color,
        opening_stats_df=df,
    )


def add_raw_scores(player_data: PlayerData, config: PipelineConfig) -> PlayerData:
    """Calculate raw performance scores: (wins + 0.5 * draws) / total_games"""
    df = player_data.opening_stats_df.copy()

    if config.verbose >= 2:
        print(f"\n{'='*80}")
        print("STEP: Calculating Raw Scores")
        print(f"{'='*80}")

    # Vectorized calculation
    df["raw_score"] = (df["num_wins"] + 0.5 * df["num_draws"]) / df["num_games"]
    df["raw_score"] = df["raw_score"].fillna(0.0).astype(float)

    # Validation
    if not df["raw_score"].between(0, 1).all():
        raise ValueError("raw_score contains values outside [0, 1]!")

    if config.verbose >= 2:
        print(f"  Range: [{df['raw_score'].min():.4f}, {df['raw_score'].max():.4f}]")
        print(f"  Mean: {df['raw_score'].mean():.4f}")

    return PlayerData(
        player_id=player_data.player_id,
        name=player_data.name,
        rating=player_data.rating,
        color=player_data.color,
        opening_stats_df=df,
    )


def remap_opening_ids(
    player_data: PlayerData,
    artifacts: PipelineArtifacts,
) -> PlayerData:
    """Remap database opening IDs to 0-based training IDs."""
    df = player_data.opening_stats_df.copy()
    config = artifacts.config

    if config.verbose >= 2:
        print(f"\n{'='*80}")
        print("STEP: Remapping Opening IDs")
        print(f"{'='*80}")

    # Vectorized mapping
    df["training_opening_id"] = df["opening_id"].map(artifacts.db_to_training_id)

    # Check for unmapped IDs (should never happen after filtering)
    if df["training_opening_id"].isna().any():
        unmapped = df.loc[df["training_opening_id"].isna(), "opening_id"].unique()
        raise ValueError(f"Found unmapped opening IDs: {unmapped}")

    df["training_opening_id"] = df["training_opening_id"].astype(int)

    if config.verbose >= 2:
        min_id = int(df["training_opening_id"].min())
        max_id = int(df["training_opening_id"].max())
        print(f"  Training ID range: [{min_id}, {max_id}]")

    return PlayerData(
        player_id=player_data.player_id,
        name=player_data.name,
        rating=player_data.rating,
        color=player_data.color,
        opening_stats_df=df,
    )


def normalize_player_rating(
    player_data: PlayerData,
    artifacts: PipelineArtifacts,
) -> PlayerData:
    """Apply z-score normalization to player rating."""
    config = artifacts.config

    if config.verbose >= 2:
        print(f"\n{'='*80}")
        print("STEP: Normalizing Player Rating")
        print(f"{'='*80}")

    if player_data.rating is None or player_data.rating <= 0:
        raise ValueError(
            f"Invalid rating for player {player_data.name}: {player_data.rating}"
        )

    rating_z = (player_data.rating - artifacts.rating_mean) / artifacts.rating_std

    if config.verbose >= 2:
        print(f"  {player_data.rating} → {rating_z:.4f}")

    if abs(rating_z) > 5:
        print(f"  ⚠️  WARNING: rating_z = {rating_z:.4f} is extreme (> 5 std devs)")

    return PlayerData(
        player_id=player_data.player_id,
        name=player_data.name,
        rating=player_data.rating,
        color=player_data.color,
        opening_stats_df=player_data.opening_stats_df,
        rating_z=rating_z,
    )


def apply_bayesian_shrinkage(
    player_data: PlayerData,
    artifacts: PipelineArtifacts,
) -> PlayerData:
    """Apply hierarchical Bayesian shrinkage toward opening-specific means."""
    df = player_data.opening_stats_df.copy()
    config = artifacts.config

    if config.verbose >= 2:
        print(f"\n{'='*80}")
        print("STEP: Applying Bayesian Shrinkage")
        print(f"{'='*80}")

    # Vectorized lookup of opening means
    def get_opening_mean(training_id: int) -> float:
        key = str(training_id)
        if key not in artifacts.opening_stats:
            raise ValueError(f"Training ID {training_id} not found in opening_stats!")
        return artifacts.opening_stats[key][0]  # Index 0 = opening_mean

    df["opening_mean"] = df["training_opening_id"].apply(get_opening_mean)

    # Bayesian shrinkage formula
    k = config.k_shrinkage
    numerator = (df["num_games"] * df["raw_score"]) + (k * df["opening_mean"])
    denominator = df["num_games"] + k

    df["adjusted_score"] = numerator / denominator
    df["confidence"] = df["num_games"] / denominator

    # Validation
    if not df["adjusted_score"].between(0, 1).all():
        raise ValueError("adjusted_score contains values outside [0, 1]!")

    if config.verbose >= 2:
        adjustment = df["adjusted_score"] - df["raw_score"]
        print(f"  Mean adjustment: {adjustment.mean():+.4f}")
        print(f"  Adjustment range: [{adjustment.min():+.4f}, {adjustment.max():+.4f}]")

    # Drop temporary column
    df = df.drop(columns=["opening_mean"])

    return PlayerData(
        player_id=player_data.player_id,
        name=player_data.name,
        rating=player_data.rating,
        color=player_data.color,
        opening_stats_df=df,
        rating_z=player_data.rating_z,
    )


def encode_eco_features(
    player_data: PlayerData,
    artifacts: PipelineArtifacts,
) -> PlayerData:
    """Encode ECO codes into categorical integer features."""
    df = player_data.opening_stats_df.copy()
    config = artifacts.config

    if config.verbose >= 2:
        print(f"\n{'='*80}")
        print("STEP: Encoding ECO Features")
        print(f"{'='*80}")

    # Parse ECO codes
    df["eco_letter_str"] = df["eco"].str[0]
    df["eco_number_str"] = df["eco"].str[1:]

    # Encode letter
    df["eco_letter_cat"] = df["eco_letter_str"].map(artifacts.eco_letter_map)
    unmapped_letters = df["eco_letter_cat"].isna()
    if unmapped_letters.any():
        if config.verbose >= 1:
            print(
                f"  ⚠️  {unmapped_letters.sum()} openings have unmapped ECO letters (using default)"
            )
        df.loc[unmapped_letters, "eco_letter_cat"] = 0
    df["eco_letter_cat"] = df["eco_letter_cat"].astype(int)

    # Encode number
    df["eco_number_cat"] = df["eco_number_str"].map(artifacts.eco_number_map)
    unmapped_numbers = df["eco_number_cat"].isna()
    if unmapped_numbers.any():
        if config.verbose >= 1:
            print(
                f"  ⚠️  {unmapped_numbers.sum()} openings have unmapped ECO numbers (using default)"
            )
        df.loc[unmapped_numbers, "eco_number_cat"] = 0
    df["eco_number_cat"] = df["eco_number_cat"].astype(int)

    # Drop temporary columns
    df = df.drop(columns=["eco_letter_str", "eco_number_str"])

    if config.verbose >= 2:
        print(f"  Encoded {len(df)} ECO codes")

    return PlayerData(
        player_id=player_data.player_id,
        name=player_data.name,
        rating=player_data.rating,
        color=player_data.color,
        opening_stats_df=df,
        rating_z=player_data.rating_z,
    )


def create_model_input(player_data: PlayerData, config: PipelineConfig) -> ModelInput:
    """Convert processed PlayerData to ModelInput (NumPy arrays)."""
    df = player_data.opening_stats_df

    if config.verbose >= 2:
        print(f"\n{'='*80}")
        print("STEP: Creating Model Input")
        print(f"{'='*80}")

    # Validation
    required_cols = {
        "training_opening_id",
        "eco_letter_cat",
        "eco_number_cat",
        "adjusted_score",
        "confidence",
        "opening_name",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if player_data.rating_z is None:
        raise ValueError("rating_z is None! Did you run normalize_player_rating?")

    # Extract to NumPy arrays
    model_input = ModelInput(
        training_player_id=None,  # Fold-in user (not in training set)
        rating_z=float(player_data.rating_z),
        opening_ids=df["training_opening_id"].to_numpy(dtype=np.int64),
        eco_letter_cats=df["eco_letter_cat"].to_numpy(dtype=np.int64),
        eco_number_cats=df["eco_number_cat"].to_numpy(dtype=np.int64),
        scores=df["adjusted_score"].to_numpy(dtype=np.float32),
        confidence=df["confidence"].to_numpy(dtype=np.float32),
        opening_names=df["opening_name"].tolist(),
    )

    if config.verbose >= 2:
        print(f"  Shape: {model_input.opening_ids.shape}")
        print(f"  Player: {player_data.name}")
        print(f"  Rating Z: {model_input.rating_z:.4f}")

    return model_input


def process_player_for_inference(
    player_data: PlayerData,
    config: PipelineConfig,
    artifacts: Optional[PipelineArtifacts] = None,
) -> ModelInput:
    """
    Main pipeline: Transform ONE player's raw data into model-ready input.

    This is the PUBLIC API for the fold-in inference pipeline.
    Call this once per user request on your web app/API.

    Args:
        player_data: Raw player data (from DB or Lichess API) for ONE player
        config: Pipeline configuration
        artifacts: Pre-loaded artifacts (if None, will load from disk)
                  In production, create artifacts once at startup and reuse.

    Returns:
        ModelInput ready for inference (for this ONE player)

    Raises:
        ValueError: If player has no valid openings or data is invalid
        FileNotFoundError: If required artifacts are missing

    Example (Production):
        >>> # At server startup - load once
        >>> config = PipelineConfig(
        ...     model_artifacts_dir=Path("data/models/20251212_152017_black"),
        ...     color='b',
        ...     verbose=0  # Silent in production
        ... )
        >>> artifacts = PipelineArtifacts(config)  # Cache this globally!
        >>>
        >>> # When user requests recommendations:
        >>> player_data = fetch_from_lichess_api(username)  # ONE player
        >>> model_input = process_player_for_inference(player_data, config, artifacts)
        >>> recommendations = run_inference(model_input)  # Send to model
        >>> return recommendations  # Return to user
    """
    # Load artifacts if not provided (cache them in production!)
    if artifacts is None:
        if config.verbose >= 1:
            print("Loading artifacts...")
        artifacts = PipelineArtifacts(config)

    if config.verbose >= 1:
        color_name = "White" if config.color == "w" else "Black"
        print(f"\n{'='*80}")
        print(f"Processing {color_name} openings for: {player_data.name}")
        print(f"{'='*80}")
        print(f"  Rating: {player_data.rating}")
        print(f"  Input openings: {len(player_data.opening_stats_df)}")
        print(f"  Total games: {player_data.total_games():,}")

    # Run pipeline
    player_data = filter_valid_openings(player_data, artifacts)
    player_data = add_raw_scores(player_data, config)
    player_data = remap_opening_ids(player_data, artifacts)
    player_data = normalize_player_rating(player_data, artifacts)
    player_data = apply_bayesian_shrinkage(player_data, artifacts)
    player_data = encode_eco_features(player_data, artifacts)
    model_input = create_model_input(player_data, config)

    if config.verbose >= 1:
        print("\n✓ Pipeline complete")
        print(f"  Output openings: {len(model_input)}")
        print("  Ready for inference")
        print(f"{'='*80}\n")

    return model_input


def run_foldin_pipeline(
    *,
    player_data: PlayerData,
    config: PipelineConfig,
    artifacts: Optional[PipelineArtifacts] = None,
) -> dict:
    """End-to-end fold-in pipeline: process player + run inference + return recommendations.

    This replaces the old behavior where the pipeline only produced `ModelInput`.
    `process_player_for_inference()` remains available as the low-level processing step.

    Returns:
        JSON-serializable dict including:
          - recommendations: list of {eco, opening_name, predicted_score}
          - stats: counts and prediction distribution
          - player: metadata (rating_z, color)

    Verbosity:
        - verbose=0: no prints
        - verbose=1: high-level progress
        - verbose=2: includes played-openings diagnostics in return payload
    """
    if artifacts is None:
        artifacts = PipelineArtifacts(config)

    model_input = process_player_for_inference(player_data, config, artifacts)

    if config.verbose >= 1:
        print("Running inference...")

    predictions = predict_all_openings(artifacts=artifacts, model_input=model_input)
    output = build_recommendations_output(
        artifacts=artifacts,
        model_input=model_input,
        predictions=predictions,
    )

    if config.verbose >= 1:
        print("✓ Inference complete")

    return output
