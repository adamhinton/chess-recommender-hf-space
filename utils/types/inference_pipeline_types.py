from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class RawOpeningStats:
    """Raw statistics for a single player-opening pair from database/API.

    This is the intermediate representation between DataFrame rows and
    processed data. Used when individual opening access is needed.

    Note: opening_id is still the database ID, not remapped to training ID yet.
    """

    opening_id: int  # Database ID (will be remapped to training ID in Step 2)
    eco: str  # ECO code
    opening_name: str  # Full opening name including ECO code (I think)
    num_games: int  # Games played
    num_wins: int  # Win count
    num_draws: int  # Draw count
    num_losses: int  # Loss count

    @property
    def raw_score(self) -> float:
        """Calculate raw performance score: (num_wins + 0.5*num_draws) / total_games.

        Returns:
            Score between 0.0 and 1.0, or 0.0 if no games played
        """
        return (
            (self.num_wins + 0.5 * self.num_draws) / self.num_games
            if self.num_games > 0
            else 0.0
        )


@dataclass
class PlayerData:
    """Complete player profile before model transformation.

    This is the main data container passed between pipeline stages.
    Keeps opening data as DataFrame for efficient vectorized operations
    while providing type-safe access to player metadata.

    The opening_stats_df should conform to OpeningStatsRow schema.
    """

    player_id: int  # Database ID (will be mapped to training ID if player exists in training set)
    name: str  # Player username
    rating: int  # Current rating for the specified color
    color: str  # 'w' for white, 'b' for black
    opening_stats_df: (
        pd.DataFrame
    )  # Opening statistics (see OpeningStatsRow for schema)
    rating_z: Optional[float] = None  # Z-score normalized rating (added in Step 2.6)

    def total_games(self) -> int:
        """Calculate total games across all openings (vectorized)."""
        return int(self.opening_stats_df["num_games"].sum())

    def total_wins(self) -> int:
        """Calculate total wins across all openings (vectorized)."""
        return int(self.opening_stats_df["num_wins"].sum())

    def mean_score(self) -> float:
        """Calculate mean performance score across all openings (vectorized)."""
        df = self.opening_stats_df
        scores = (df["num_wins"] + 0.5 * df["num_draws"]) / df["num_games"]
        return float(scores.mean())

    def filter_by_games(self, min_games: int) -> pd.DataFrame:
        """Filter openings by minimum game threshold.

        Args:
            min_games: Minimum number of games required

        Returns:
            Filtered DataFrame copy
        """
        return self.opening_stats_df[
            self.opening_stats_df["num_games"] >= min_games
        ].copy()

    def get_opening_stats(self, opening_id: int) -> RawOpeningStats:
        """Get statistics for a specific opening as dataclass.

        Args:
            opening_id: Database opening ID

        Returns:
            RawOpeningStats object for the specified opening

        Raises:
            IndexError: If opening_id not found
        """
        row = self.opening_stats_df[
            self.opening_stats_df["opening_id"] == opening_id
        ].iloc[0]
        return RawOpeningStats(
            opening_id=int(row["opening_id"]),
            eco=row["eco"],
            opening_name=row["opening_name"],
            num_games=int(row["num_games"]),
            num_wins=int(row["num_wins"]),
            num_draws=int(row["num_draws"]),
            num_losses=int(row["num_losses"]),
        )

@dataclass
class ModelInput:
    """Final data structure ready for HuggingFace model inference.

    This is what gets sent to the model API. All player and opening data
    are converted to parallel numpy arrays for efficient batch processing.

    For fold-in (new) users, training_player_id is None and the model uses
    only rating_z and opening features to make predictions.

    Arrays have parallel indices: opening_ids[i] corresponds to eco_letter_cats[i],
    eco_number_cats[i], scores[i], and confidence[i].
    """

    # Player features
    training_player_id: Optional[int]  # None for fold-in users, int for known users
    rating_z: float  # Z-score normalized rating

    # Opening features (parallel arrays, length N = number of openings)
    opening_ids: np.ndarray  # int64, shape (N,) - training opening IDs
    eco_letter_cats: np.ndarray  # int64, shape (N,) - ECO letter categories (0-4)
    eco_number_cats: np.ndarray  # int64, shape (N,) - ECO numbers (0-99)
    scores: np.ndarray  # float32, shape (N,) - adjusted scores
    confidence: np.ndarray  # float32, shape (N,) - confidence weights

    # Metadata (not sent to model, used for post-processing)
    opening_names: List[str]  # Opening names in same order as arrays

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict for HuggingFace API.

        Returns:
            Dictionary with all fields converted to Python native types
            (numpy arrays -> lists, numpy scalars -> Python scalars)
        """
        return {
            "player_id": self.training_player_id,  # None for fold-in
            "rating_z": float(self.rating_z),
            "opening_ids": self.opening_ids.tolist(),
            "eco_letter_cats": self.eco_letter_cats.tolist(),
            "eco_number_cats": self.eco_number_cats.tolist(),
            "scores": self.scores.tolist(),
            "confidence": self.confidence.tolist(),
        }

    def __len__(self) -> int:
        """Return number of openings in this input."""
        return len(self.opening_ids)
