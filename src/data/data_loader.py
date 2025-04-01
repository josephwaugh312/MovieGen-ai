"""
Data loader module for the MovieLens dataset.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"

def load_movielens_100k() -> Dict[str, pd.DataFrame]:
    """
    Load the MovieLens 100K dataset.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for different parts of the dataset
    """
    dataset_path = RAW_DATA_DIR / "ml-100k"
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please run the download_data.py script first."
        )
    
    logger.info("Loading MovieLens 100K dataset")
    
    # Define column names for each dataset file
    column_names = {
        "users": ["user_id", "age", "gender", "occupation", "zip_code"],
        "items": ["movie_id", "title", "release_date", "video_release_date", 
                  "imdb_url", "unknown", "action", "adventure", "animation", 
                  "children", "comedy", "crime", "documentary", "drama", "fantasy", 
                  "film_noir", "horror", "musical", "mystery", "romance", "sci_fi", 
                  "thriller", "war", "western"],
        "ratings": ["user_id", "movie_id", "rating", "timestamp"],
    }

    # Load the data files
    data = {}
    
    # Load user data
    data["users"] = pd.read_csv(
        dataset_path / "u.user", 
        sep="|", 
        names=column_names["users"],
        encoding="latin-1"
    )
    
    # Load movie data
    data["movies"] = pd.read_csv(
        dataset_path / "u.item", 
        sep="|", 
        names=column_names["items"],
        encoding="latin-1"
    )
    
    # Load ratings data
    data["ratings"] = pd.read_csv(
        dataset_path / "u.data", 
        sep="\t", 
        names=column_names["ratings"],
        encoding="latin-1"
    )
    
    # Transform timestamp to datetime
    data["ratings"]["timestamp"] = pd.to_datetime(data["ratings"]["timestamp"], unit="s")

    # Extract genre columns
    genre_columns = column_names["items"][5:]
    data["genres"] = data["movies"][["movie_id"] + genre_columns]
    
    # Remove genre columns from movies dataframe to simplify
    data["movies"] = data["movies"].drop(columns=genre_columns)
    
    logger.info(f"Loaded users: {len(data['users'])} rows")
    logger.info(f"Loaded movies: {len(data['movies'])} rows")
    logger.info(f"Loaded ratings: {len(data['ratings'])} rows")
    
    return data

def split_train_test(
    ratings: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings data into training and test sets.
    
    Args:
        ratings (pd.DataFrame): Ratings DataFrame
        test_size (float): Fraction of data to use for testing
        random_state (int, optional): Random seed for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_ratings, test_ratings)
    """
    # Shuffle the data
    shuffled_ratings = ratings.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split point
    split_point = int(len(shuffled_ratings) * (1 - test_size))
    
    # Split the data
    train_ratings = shuffled_ratings.iloc[:split_point].copy()
    test_ratings = shuffled_ratings.iloc[split_point:].copy()
    
    logger.info(f"Split data into {len(train_ratings)} train and {len(test_ratings)} test samples")
    
    return train_ratings, test_ratings

def get_movie_genres(movies_df: pd.DataFrame, genres_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame mapping movies to their genres as a list.
    
    Args:
        movies_df (pd.DataFrame): Movies DataFrame
        genres_df (pd.DataFrame): Genres DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with movie_id and genres columns
    """
    # Get genre columns
    genre_columns = [col for col in genres_df.columns if col != "movie_id"]
    
    # Create a new DataFrame
    movie_genres = []
    
    for _, row in genres_df.iterrows():
        movie_id = row["movie_id"]
        # Get genres that have a value of 1
        genres = [genre for genre in genre_columns if row[genre] == 1]
        movie_genres.append({"movie_id": movie_id, "genres": genres})
    
    return pd.DataFrame(movie_genres)

if __name__ == "__main__":
    # Load the dataset
    data = load_movielens_100k()
    
    # Example: create train/test split
    train_ratings, test_ratings = split_train_test(data["ratings"])
    
    # Example: get movie genres
    movie_genres = get_movie_genres(data["movies"], data["genres"])
    print(movie_genres.head())
    
    # Example: exploring the data
    print("\nRatings distribution:")
    print(data["ratings"]["rating"].value_counts().sort_index())
    
    print("\nUser age distribution:")
    print(data["users"]["age"].describe()) 