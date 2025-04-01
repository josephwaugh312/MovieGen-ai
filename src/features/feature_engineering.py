"""
Feature engineering module for the movie recommendation system.
This module provides functions to transform raw data into features for recommendation models.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_user_item_matrix(ratings_df: pd.DataFrame) -> sp.csr_matrix:
    """
    Create a sparse user-item matrix from ratings dataframe.
    
    Args:
        ratings_df: DataFrame containing user_id, movie_id, and rating columns
        
    Returns:
        sp.csr_matrix: Sparse matrix of user-item ratings
        dict: Mapping of matrix row indices to user IDs
        dict: Mapping of matrix column indices to movie IDs
    """
    # Get unique users and items
    users = ratings_df['user_id'].unique()
    movies = ratings_df['movie_id'].unique()
    
    # Create mappings from IDs to indices
    user_to_idx = {user: i for i, user in enumerate(users)}
    movie_to_idx = {movie: i for i, movie in enumerate(movies)}
    
    # Create reverse mappings (indices to IDs)
    idx_to_user = {i: user for user, i in user_to_idx.items()}
    idx_to_movie = {i: movie for movie, i in movie_to_idx.items()}
    
    # Map user and movie IDs to matrix indices
    user_indices = ratings_df['user_id'].map(user_to_idx).values
    movie_indices = ratings_df['movie_id'].map(movie_to_idx).values
    ratings = ratings_df['rating'].values
    
    # Create sparse matrix
    shape = (len(users), len(movies))
    rating_matrix = sp.csr_matrix((ratings, (user_indices, movie_indices)), shape=shape)
    
    logger.info(f"Created user-item matrix of shape {shape} with {len(ratings)} non-zero entries")
    logger.info(f"Matrix density: {len(ratings) / (shape[0] * shape[1]):.4f}")
    
    return rating_matrix, idx_to_user, idx_to_movie, user_to_idx, movie_to_idx

def create_movie_feature_matrix(movies_df: pd.DataFrame, genres_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Create a feature matrix for movies based on genres.
    
    Args:
        movies_df: DataFrame containing movie metadata
        genres_df: DataFrame containing movie genre binary indicators
        
    Returns:
        np.ndarray: Feature matrix where each row is a movie
        List[str]: List of feature names
    """
    # Extract genre columns (remove movie_id)
    genre_cols = [col for col in genres_df.columns if col != 'movie_id']
    
    # Sort movies by movie_id for consistency
    genres_df = genres_df.sort_values('movie_id')
    
    # Create feature matrix (just use genres as features for now)
    feature_matrix = genres_df[genre_cols].values
    
    logger.info(f"Created movie feature matrix of shape {feature_matrix.shape}")
    
    return feature_matrix, genre_cols

def create_user_feature_matrix(users_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Create a feature matrix for users based on demographics.
    
    Args:
        users_df: DataFrame containing user metadata
        
    Returns:
        np.ndarray: Feature matrix where each row is a user
        List[str]: List of feature names
    """
    # Sort users by user_id for consistency
    users_df = users_df.sort_values('user_id')
    
    # Create age groups
    users_df['age_group'] = pd.cut(
        users_df['age'], 
        bins=[0, 18, 25, 35, 45, 55, 100],
        labels=['<18', '18-24', '25-34', '35-44', '45-54', '55+']
    )
    
    # One-hot encode categorical variables
    feature_df = pd.get_dummies(
        users_df[['gender', 'age_group', 'occupation']],
        columns=['gender', 'age_group', 'occupation'],
        prefix=['gender', 'age', 'job']
    )
    
    # Create feature matrix and get feature names
    feature_matrix = feature_df.values
    feature_names = feature_df.columns.tolist()
    
    logger.info(f"Created user feature matrix of shape {feature_matrix.shape}")
    
    return feature_matrix, feature_names

def normalize_matrix(matrix: np.ndarray, axis=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize a matrix by subtracting mean and optionally dividing by std along axis.
    
    Args:
        matrix: Input matrix to normalize
        axis: Axis along which to normalize (0=columns, 1=rows)
        
    Returns:
        np.ndarray: Normalized matrix
        np.ndarray: Means used for normalization
        np.ndarray: Standard deviations used for normalization
    """
    # Calculate mean and std along specified axis
    means = np.nanmean(matrix, axis=axis, keepdims=True)
    stds = np.nanstd(matrix, axis=axis, keepdims=True)
    
    # Replace zero standard deviations with 1 to avoid division by zero
    stds[stds == 0] = 1
    
    # Normalize the matrix
    normalized = (matrix - means) / stds
    
    # If axis=1, means and stds are shape (n, 1), convert to 1D
    if axis == 1:
        means = means.ravel()
        stds = stds.ravel()
    elif axis == 0:
        means = means.ravel()
        stds = stds.ravel()
    
    return normalized, means, stds

def split_train_test_by_time(
    ratings_df: pd.DataFrame, 
    test_size: float = 0.2,
    timestamp_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and test sets based on timestamp.
    
    Args:
        ratings_df: DataFrame containing ratings
        test_size: Fraction of data to use for testing
        timestamp_col: Name of the column containing timestamp
        
    Returns:
        pd.DataFrame: Training set
        pd.DataFrame: Test set
    """
    # Sort by timestamp
    sorted_data = ratings_df.sort_values(timestamp_col)
    
    # Determine split point
    split_idx = int(len(sorted_data) * (1 - test_size))
    
    # Split the data
    train = sorted_data.iloc[:split_idx].copy()
    test = sorted_data.iloc[split_idx:].copy()
    
    logger.info(f"Split data into {len(train)} train and {len(test)} test instances based on time")
    
    # Log split timestamp info
    if isinstance(train[timestamp_col].iloc[-1], pd.Timestamp) and isinstance(test[timestamp_col].iloc[0], pd.Timestamp):
        split_time = train[timestamp_col].iloc[-1]
        logger.info(f"Split date: train data <= {split_time}, test data > {split_time}")
    
    return train, test

def create_movie_user_feature_map(
    rating_matrix: sp.csr_matrix, 
    movie_features: np.ndarray, 
    movie_to_idx: Dict[int, int]
) -> Dict[int, np.ndarray]:
    """
    Create a mapping from user indices to movie features they've rated.
    
    Args:
        rating_matrix: User-item rating matrix
        movie_features: Matrix of movie features
        movie_to_idx: Mapping from movie IDs to matrix indices
        
    Returns:
        Dict[int, np.ndarray]: Map from user index to average movie features
    """
    user_to_feature_map = {}
    
    # For each user (row in rating matrix)
    for user_idx in range(rating_matrix.shape[0]):
        # Get indices of movies rated by this user
        _, movie_indices = rating_matrix[user_idx].nonzero()
        
        if len(movie_indices) > 0:
            # Get features of rated movies and average them
            user_movie_features = movie_features[movie_indices]
            avg_features = np.mean(user_movie_features, axis=0)
            user_to_feature_map[user_idx] = avg_features
    
    logger.info(f"Created user-movie feature map for {len(user_to_feature_map)} users")
    
    return user_to_feature_map

if __name__ == "__main__":
    # Example usage with MovieLens dataset
    from src.data.data_loader import load_movielens_100k
    
    # Load data
    data = load_movielens_100k()
    
    # Create user-item matrix
    rating_matrix, idx_to_user, idx_to_movie, user_to_idx, movie_to_idx = create_user_item_matrix(data["ratings"])
    
    # Create movie features
    movie_features, feature_names = create_movie_feature_matrix(data["movies"], data["genres"])
    
    # Create user features
    user_features, user_feature_names = create_user_feature_matrix(data["users"])
    
    # Create time-based train/test split
    train_ratings, test_ratings = split_train_test_by_time(data["ratings"])
    
    # Print some statistics
    print(f"Rating matrix shape: {rating_matrix.shape}")
    print(f"Movie features shape: {movie_features.shape}")
    print(f"User features shape: {user_features.shape}")
    print(f"Train set size: {len(train_ratings)}")
    print(f"Test set size: {len(test_ratings)}")
    
    # Create content-based user profiles
    user_movie_profiles = create_movie_user_feature_map(rating_matrix, movie_features, movie_to_idx)
    print(f"User movie profiles created for {len(user_movie_profiles)} users") 