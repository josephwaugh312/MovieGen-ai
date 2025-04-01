"""
Collaborative Filtering module for recommendation systems.
Contains implementations of item-based and user-based collaborative filtering.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cosine
from typing import Dict, Tuple, List, Optional, Union
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_similarity_matrix(
    matrix: sp.csr_matrix, 
    method: str = 'cosine',
    min_support: int = 5,
    is_user_similarity: bool = False
) -> np.ndarray:
    """
    Compute similarity matrix between items or users.
    
    Args:
        matrix: User-item rating matrix (users as rows, items as columns)
        method: Similarity method ('cosine', 'pearson', 'adjusted_cosine')
        min_support: Minimum number of shared ratings to compute similarity
        is_user_similarity: If True, compute user-user similarity; if False, item-item
        
    Returns:
        np.ndarray: Similarity matrix
    """
    if not is_user_similarity:
        # For item-item similarity, we need items as rows
        matrix = matrix.T.tocsr()
    
    n_entities = matrix.shape[0]  # Number of items/users
    sim_matrix = np.zeros((n_entities, n_entities))
    
    # For adjusted cosine, normalize ratings by user mean
    if method == 'adjusted_cosine' and not is_user_similarity:
        # Calculate user means
        user_means = np.zeros(matrix.shape[1])
        for i in range(matrix.shape[1]):
            if matrix[:, i].nnz > 0:
                user_means[i] = matrix[:, i].data.mean()
        
        # Subtract user means from ratings
        for i in range(matrix.shape[0]):
            row_indices = matrix.getrow(i).indices
            if len(row_indices) > 0:
                row_data = matrix.getrow(i).data
                matrix.data[matrix.indptr[i]:matrix.indptr[i+1]] = row_data - user_means[row_indices]
    
    # Compute similarities
    start_time = time.time()
    logger.info(f"Computing {method} similarities for {n_entities} entities")
    
    # For each entity
    for i in range(n_entities):
        # Progress logging for large matrices
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / i) * (n_entities - i)
            logger.info(f"Processed {i}/{n_entities} entities. Time: {elapsed:.2f}s, Est. remaining: {remaining:.2f}s")
        
        # Entity i's ratings
        i_ratings = matrix.getrow(i)
        i_indices = i_ratings.indices
        
        # For each other entity
        for j in range(i, n_entities):  # Start from i to exploit symmetry
            # Entity j's ratings
            j_ratings = matrix.getrow(j)
            j_indices = j_ratings.indices
            
            # Find common rated items/users
            common_indices = np.intersect1d(i_indices, j_indices)
            
            # Skip if too few common ratings
            if len(common_indices) < min_support:
                sim_matrix[i, j] = sim_matrix[j, i] = 0
                continue
            
            # Extract ratings for common items
            i_data = np.array([i_ratings[0, idx] for idx in common_indices])
            j_data = np.array([j_ratings[0, idx] for idx in common_indices])
            
            # Compute similarity based on selected method
            if method == 'cosine' or method == 'adjusted_cosine':
                # Cosine similarity: 1 - cosine distance
                similarity = 1 - cosine(i_data, j_data)
            elif method == 'pearson':
                # Pearson correlation
                i_mean = np.mean(i_data)
                j_mean = np.mean(j_data)
                numerator = np.sum((i_data - i_mean) * (j_data - j_mean))
                denominator = np.sqrt(np.sum((i_data - i_mean)**2) * np.sum((j_data - j_mean)**2))
                similarity = numerator / denominator if denominator != 0 else 0
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            # Handle NaN similarity
            if np.isnan(similarity):
                similarity = 0
            
            # Store symmetrically
            sim_matrix[i, j] = sim_matrix[j, i] = similarity
    
    elapsed = time.time() - start_time
    logger.info(f"Similarity computation completed in {elapsed:.2f} seconds")
    
    return sim_matrix

class ItemBasedCF:
    """Item-based collaborative filtering recommender."""
    
    def __init__(
        self, 
        similarity_method: str = 'cosine',
        min_support: int = 5,
        k_neighbors: int = 50
    ):
        """
        Initialize item-based CF recommender.
        
        Args:
            similarity_method: Method to compute item similarities
            min_support: Minimum number of users who rated both items
            k_neighbors: Number of similar items to consider for predictions
        """
        self.similarity_method = similarity_method
        self.min_support = min_support
        self.k_neighbors = k_neighbors
        self.sim_matrix = None
        self.rating_matrix = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.user_means = None
        
        logger.info(f"Initialized ItemBasedCF with {similarity_method} similarity, "
                   f"min_support={min_support}, k_neighbors={k_neighbors}")
    
    def fit(
        self, 
        rating_matrix: sp.csr_matrix,
        idx_to_user: Dict[int, int],
        idx_to_movie: Dict[int, int],
        user_to_idx: Dict[int, int],
        movie_to_idx: Dict[int, int]
    ):
        """
        Fit the model to training data.
        
        Args:
            rating_matrix: User-item rating matrix
            idx_to_user: Mapping from matrix row indices to user IDs
            idx_to_movie: Mapping from matrix column indices to movie IDs
            user_to_idx: Mapping from user IDs to matrix row indices
            movie_to_idx: Mapping from movie IDs to matrix column indices
        """
        self.rating_matrix = rating_matrix
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        self.user_to_idx = user_to_idx
        self.movie_to_idx = movie_to_idx
        
        # Compute user means for adjusted predictions
        self.user_means = np.zeros(rating_matrix.shape[0])
        for i in range(rating_matrix.shape[0]):
            if rating_matrix[i].nnz > 0:
                self.user_means[i] = rating_matrix[i].data.mean()
        
        # Compute item similarities
        logger.info("Computing item similarity matrix")
        self.sim_matrix = compute_similarity_matrix(
            rating_matrix, 
            method=self.similarity_method,
            min_support=self.min_support,
            is_user_similarity=False
        )
        
        logger.info("Item-based CF model fitted")
        return self
    
    def predict(self, user_idx: int, movie_idx: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_idx: User matrix index
            movie_idx: Movie matrix index
            
        Returns:
            float: Predicted rating
        """
        if self.sim_matrix is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get user's ratings
        user_ratings = self.rating_matrix[user_idx].toarray().ravel()
        
        # If user has already rated this movie, return the actual rating
        if user_ratings[movie_idx] > 0:
            return user_ratings[movie_idx]
        
        # Get item similarities to target item
        item_sims = self.sim_matrix[movie_idx]
        
        # Get indices of items the user has rated
        rated_items = np.where(user_ratings > 0)[0]
        
        # If user hasn't rated any item, return user mean
        if len(rated_items) == 0:
            return self.user_means[user_idx] if self.user_means[user_idx] > 0 else 3.0
        
        # Get similarities between target item and items the user has rated
        sims = item_sims[rated_items]
        
        # Get user's ratings for these items
        ratings = user_ratings[rated_items]
        
        # Select top-k neighbors
        if len(sims) > self.k_neighbors:
            top_k_idx = np.argsort(sims)[-self.k_neighbors:]
            sims = sims[top_k_idx]
            ratings = ratings[top_k_idx]
        
        # If no similar items, return user mean
        if len(sims) == 0 or np.sum(np.abs(sims)) == 0:
            return self.user_means[user_idx] if self.user_means[user_idx] > 0 else 3.0
        
        # Weighted average of ratings
        if self.similarity_method in ['cosine', 'adjusted_cosine']:
            # For cosine similarity
            pred = np.sum(sims * ratings) / np.sum(np.abs(sims))
        else:
            # For pearson correlation, use baseline (user mean)
            user_mean = self.user_means[user_idx]
            pred = user_mean + np.sum(sims * (ratings - user_mean)) / np.sum(np.abs(sims))
        
        # Clip prediction to rating range [1, 5]
        pred = max(1.0, min(5.0, pred))
        
        return pred
    
    def predict_for_user(self, user_idx: int, movie_indices: List[int] = None) -> np.ndarray:
        """
        Predict ratings for a user across all or specified items.
        
        Args:
            user_idx: User matrix index
            movie_indices: List of movie indices to predict for (None = all)
            
        Returns:
            np.ndarray: Array of predicted ratings
        """
        if movie_indices is None:
            movie_indices = list(range(self.rating_matrix.shape[1]))
        
        predictions = np.zeros(len(movie_indices))
        for i, movie_idx in enumerate(movie_indices):
            predictions[i] = self.predict(user_idx, movie_idx)
        
        return predictions
    
    def recommend_for_user(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get already rated items
        user_ratings = self.rating_matrix[user_idx].toarray().ravel()
        rated_mask = user_ratings > 0 if exclude_rated else np.zeros_like(user_ratings, dtype=bool)
        
        # Candidate items = unrated items
        candidate_indices = np.where(~rated_mask)[0]
        
        # If no candidates, return empty list
        if len(candidate_indices) == 0:
            return []
        
        # Predict ratings for candidate items
        predictions = self.predict_for_user(user_idx, candidate_indices)
        
        # Sort by predicted rating
        sorted_indices = np.argsort(predictions)[::-1][:n_recommendations]
        
        # Return top-n recommendations
        recommendations = [
            (self.idx_to_movie[candidate_indices[idx]], predictions[idx])
            for idx in sorted_indices
        ]
        
        return recommendations

class UserBasedCF:
    """User-based collaborative filtering recommender."""
    
    def __init__(
        self, 
        similarity_method: str = 'cosine',
        min_support: int = 5,
        k_neighbors: int = 50
    ):
        """
        Initialize user-based CF recommender.
        
        Args:
            similarity_method: Method to compute user similarities
            min_support: Minimum number of items rated by both users
            k_neighbors: Number of similar users to consider for predictions
        """
        self.similarity_method = similarity_method
        self.min_support = min_support
        self.k_neighbors = k_neighbors
        self.sim_matrix = None
        self.rating_matrix = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.user_means = None
        
        logger.info(f"Initialized UserBasedCF with {similarity_method} similarity, "
                   f"min_support={min_support}, k_neighbors={k_neighbors}")
    
    def fit(
        self, 
        rating_matrix: sp.csr_matrix,
        idx_to_user: Dict[int, int],
        idx_to_movie: Dict[int, int],
        user_to_idx: Dict[int, int],
        movie_to_idx: Dict[int, int]
    ):
        """
        Fit the model to training data.
        
        Args:
            rating_matrix: User-item rating matrix
            idx_to_user: Mapping from matrix row indices to user IDs
            idx_to_movie: Mapping from matrix column indices to movie IDs
            user_to_idx: Mapping from user IDs to matrix row indices
            movie_to_idx: Mapping from movie IDs to matrix column indices
        """
        self.rating_matrix = rating_matrix
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        self.user_to_idx = user_to_idx
        self.movie_to_idx = movie_to_idx
        
        # Compute user means for predictions
        self.user_means = np.zeros(rating_matrix.shape[0])
        for i in range(rating_matrix.shape[0]):
            if rating_matrix[i].nnz > 0:
                self.user_means[i] = rating_matrix[i].data.mean()
        
        # Compute user similarities
        logger.info("Computing user similarity matrix")
        self.sim_matrix = compute_similarity_matrix(
            rating_matrix, 
            method=self.similarity_method,
            min_support=self.min_support,
            is_user_similarity=True
        )
        
        logger.info("User-based CF model fitted")
        return self
    
    def predict(self, user_idx: int, movie_idx: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_idx: User matrix index
            movie_idx: Movie matrix index
            
        Returns:
            float: Predicted rating
        """
        if self.sim_matrix is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get user's ratings
        user_ratings = self.rating_matrix[user_idx].toarray().ravel()
        
        # If user has already rated this movie, return the actual rating
        if user_ratings[movie_idx] > 0:
            return user_ratings[movie_idx]
        
        # Get user similarities to target user
        user_sims = self.sim_matrix[user_idx]
        
        # Get ratings for this movie from all users
        movie_ratings = self.rating_matrix[:, movie_idx].toarray().ravel()
        
        # Find users who rated this movie
        rated_users = np.where(movie_ratings > 0)[0]
        
        # If no one rated this movie, return user mean
        if len(rated_users) == 0:
            return self.user_means[user_idx] if self.user_means[user_idx] > 0 else 3.0
        
        # Get similarities between target user and users who rated this movie
        sims = user_sims[rated_users]
        
        # Get these users' ratings for this movie
        ratings = movie_ratings[rated_users]
        
        # Select top-k neighbors
        if len(sims) > self.k_neighbors:
            top_k_idx = np.argsort(sims)[-self.k_neighbors:]
            sims = sims[top_k_idx]
            ratings = ratings[top_k_idx]
        
        # If no similar users, return user mean
        if len(sims) == 0 or np.sum(np.abs(sims)) == 0:
            return self.user_means[user_idx] if self.user_means[user_idx] > 0 else 3.0
        
        # Weighted average of ratings (with or without baseline adjustment)
        if self.similarity_method == 'cosine':
            # For cosine similarity, plain weighted average
            pred = np.sum(sims * ratings) / np.sum(np.abs(sims))
        else:
            # For pearson correlation, use baseline adjustment
            user_mean = self.user_means[user_idx]
            # Get means of similar users
            sim_user_means = np.array([self.user_means[rated_users[i]] for i in range(len(ratings))])
            # Adjust ratings by user means
            pred = user_mean + np.sum(sims * (ratings - sim_user_means)) / np.sum(np.abs(sims))
        
        # Clip prediction to rating range [1, 5]
        pred = max(1.0, min(5.0, pred))
        
        return pred
    
    def predict_for_user(self, user_idx: int, movie_indices: List[int] = None) -> np.ndarray:
        """
        Predict ratings for a user across all or specified items.
        
        Args:
            user_idx: User matrix index
            movie_indices: List of movie indices to predict for (None = all)
            
        Returns:
            np.ndarray: Array of predicted ratings
        """
        if movie_indices is None:
            movie_indices = list(range(self.rating_matrix.shape[1]))
        
        predictions = np.zeros(len(movie_indices))
        for i, movie_idx in enumerate(movie_indices):
            predictions[i] = self.predict(user_idx, movie_idx)
        
        return predictions
    
    def recommend_for_user(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get already rated items
        user_ratings = self.rating_matrix[user_idx].toarray().ravel()
        rated_mask = user_ratings > 0 if exclude_rated else np.zeros_like(user_ratings, dtype=bool)
        
        # Candidate items = unrated items
        candidate_indices = np.where(~rated_mask)[0]
        
        # If no candidates, return empty list
        if len(candidate_indices) == 0:
            return []
        
        # Predict ratings for candidate items
        predictions = self.predict_for_user(user_idx, candidate_indices)
        
        # Sort by predicted rating
        sorted_indices = np.argsort(predictions)[::-1][:n_recommendations]
        
        # Return top-n recommendations
        recommendations = [
            (self.idx_to_movie[candidate_indices[idx]], predictions[idx])
            for idx in sorted_indices
        ]
        
        return recommendations

if __name__ == "__main__":
    # Example usage with MovieLens dataset
    from src.data.data_loader import load_movielens_100k
    from src.features.feature_engineering import create_user_item_matrix
    
    # Load data
    data = load_movielens_100k()
    
    # Create user-item matrix
    logger.info("Creating user-item matrix")
    rating_matrix, idx_to_user, idx_to_movie, user_to_idx, movie_to_idx = create_user_item_matrix(data["ratings"])
    
    # Create and fit item-based CF model
    logger.info("Creating and fitting item-based CF model")
    item_cf = ItemBasedCF(similarity_method='cosine', min_support=3, k_neighbors=30)
    item_cf.fit(rating_matrix, idx_to_user, idx_to_movie, user_to_idx, movie_to_idx)
    
    # Generate recommendations for a random user
    test_user_id = 1  # Replace with a user ID from your dataset
    logger.info(f"Generating recommendations for user {test_user_id}")
    recommendations = item_cf.recommend_for_user(test_user_id, n_recommendations=5)
    
    # Display recommendations
    print(f"Top 5 recommendations for user {test_user_id}:")
    for movie_id, predicted_rating in recommendations:
        movie_title = data["movies"][data["movies"]["movie_id"] == movie_id]["title"].values[0]
        print(f"  - {movie_title} (predicted rating: {predicted_rating:.2f})")
    
    # Optional: Create and fit user-based CF model
    # This is slower so commented out by default
    # logger.info("Creating and fitting user-based CF model")
    # user_cf = UserBasedCF(similarity_method='cosine', min_support=5, k_neighbors=30)
    # user_cf.fit(rating_matrix, idx_to_user, idx_to_movie, user_to_idx, movie_to_idx) 