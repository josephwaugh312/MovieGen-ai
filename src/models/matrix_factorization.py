"""
Matrix Factorization module for recommendation systems.
Contains implementations of SVD and ALS algorithms.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import optimize
from typing import Dict, Tuple, List, Optional, Union
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MatrixFactorization:
    """Base class for matrix factorization algorithms."""
    
    def __init__(
        self, 
        n_factors: int = 50,
        regularization: float = 0.1,
        learning_rate: Optional[float] = None,
        n_iterations: int = 20,
        init_mean: float = 0.0,
        init_std: float = 0.1,
        method: str = 'als'
    ):
        """
        Initialize matrix factorization model.
        
        Args:
            n_factors: Number of latent factors
            regularization: Regularization parameter
            learning_rate: Learning rate for SGD (not used in ALS)
            n_iterations: Number of iterations to run
            init_mean: Mean of initial latent factors
            init_std: Standard deviation of initial latent factors
            method: 'als' or 'sgd'
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.init_mean = init_mean
        self.init_std = init_std
        self.method = method
        
        # Model parameters
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
        
        # Mappings
        self.idx_to_user = None
        self.idx_to_movie = None
        self.user_to_idx = None
        self.movie_to_idx = None
        
        logger.info(f"Initialized MatrixFactorization with method={method}, "
                   f"n_factors={n_factors}, regularization={regularization}")
    
    def _init_model(self, n_users: int, n_items: int):
        """
        Initialize model parameters.
        
        Args:
            n_users: Number of users
            n_items: Number of items
        """
        # Initialize latent factors
        self.user_factors = self.init_mean + self.init_std * np.random.randn(n_users, self.n_factors)
        self.item_factors = self.init_mean + self.init_std * np.random.randn(n_items, self.n_factors)
        
        # Initialize biases
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = 0.0
    
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
        self.idx_to_user = idx_to_user
        self.idx_to_movie = idx_to_movie
        self.user_to_idx = user_to_idx
        self.movie_to_idx = movie_to_idx
        
        # Calculate global bias (mean of all ratings)
        self.global_bias = rating_matrix.data.mean()
        
        # Initialize model parameters
        n_users, n_items = rating_matrix.shape
        self._init_model(n_users, n_items)
        
        if self.method == 'als':
            self._fit_als(rating_matrix)
        elif self.method == 'sgd':
            self._fit_sgd(rating_matrix)
        else:
            raise ValueError(f"Unknown method: {self.method}. Expected 'als' or 'sgd'.")
        
        return self
    
    def _fit_als(self, rating_matrix: sp.csr_matrix):
        """
        Fit the model using Alternating Least Squares.
        
        Args:
            rating_matrix: User-item rating matrix
        """
        n_users, n_items = rating_matrix.shape
        
        # Initialize user and item factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Compute global bias (average rating)
        self.global_bias = rating_matrix.data.mean()
        
        # Create a copy of the rating matrix for centering
        rating_matrix_copy = rating_matrix.copy()
        
        # Ensure the data is float64 before subtracting the global bias
        rating_matrix_copy.data = rating_matrix_copy.data.astype(np.float64)
        rating_matrix_copy.data -= self.global_bias
        
        # Convert to COO format for ALS
        coo_matrix = rating_matrix_copy.tocoo()
        
        logger.info(f"Starting ALS with {self.n_iterations} iterations")
        
        # Run ALS iterations
        for iteration in range(self.n_iterations):
            start_time = time.time()
            
            # Update user factors
            for u in range(n_users):
                # Get items rated by user u
                item_indices = rating_matrix[u].indices
                
                if len(item_indices) == 0:
                    continue
                
                # Get ratings
                ratings = rating_matrix_copy[u, item_indices].toarray().ravel()
                
                # Get corresponding item factors
                factors = self.item_factors[item_indices]
                
                # Calculate A and b for least squares
                A = factors.T @ factors + self.regularization * np.eye(self.n_factors)
                b = factors.T @ ratings
                
                # Update user factors
                self.user_factors[u] = np.linalg.solve(A, b)
            
            # Update item factors
            for i in range(n_items):
                # Get users who rated item i
                user_indices = rating_matrix[:, i].nonzero()[0]
                
                if len(user_indices) == 0:
                    continue
                
                # Get ratings
                ratings = rating_matrix_copy[user_indices, i].toarray().ravel()
                
                # Get corresponding user factors
                factors = self.user_factors[user_indices]
                
                # Calculate A and b for least squares
                A = factors.T @ factors + self.regularization * np.eye(self.n_factors)
                b = factors.T @ ratings
                
                # Update item factors
                self.item_factors[i] = np.linalg.solve(A, b)
            
            # Calculate RMSE on training data (optional)
            if iteration % 5 == 0 or iteration == self.n_iterations - 1:
                predictions = self.global_bias + self.user_factors @ self.item_factors.T
                rmse = self._calculate_rmse(coo_matrix, predictions)
                elapsed_time = time.time() - start_time
                logger.info(f"Iteration {iteration+1}/{self.n_iterations}: RMSE = {rmse:.4f}, Time: {elapsed_time:.2f}s")
        
        logger.info(f"ALS completed in {self.n_iterations} iterations")
        return self
    
    def _fit_sgd(self, rating_matrix: sp.csr_matrix):
        """
        Fit the model using Stochastic Gradient Descent.
        
        Args:
            rating_matrix: User-item rating matrix
        """
        if self.learning_rate is None:
            self.learning_rate = 0.005
        
        # Convert sparse matrix to COO format for easier iteration
        rating_matrix_coo = rating_matrix.tocoo()
        
        logger.info(f"Starting SGD optimization for {self.n_iterations} iterations")
        start_time = time.time()
        
        n_ratings = len(rating_matrix_coo.data)
        
        for iteration in range(self.n_iterations):
            iter_start = time.time()
            
            # Shuffle indices for SGD
            indices = np.arange(n_ratings)
            np.random.shuffle(indices)
            
            # Process batches
            batch_size = 1000
            n_batches = (n_ratings + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, n_ratings)
                batch_indices = indices[batch_start:batch_end]
                
                users = rating_matrix_coo.row[batch_indices]
                items = rating_matrix_coo.col[batch_indices]
                ratings = rating_matrix_coo.data[batch_indices]
                
                # Update model parameters for each rating in the batch
                for u, i, r in zip(users, items, ratings):
                    # Predict rating
                    pred = self.global_bias + self.user_biases[u] + self.item_biases[i] + \
                           np.dot(self.user_factors[u], self.item_factors[i])
                    
                    # Calculate error
                    error = r - pred
                    
                    # Update biases
                    self.user_biases[u] += self.learning_rate * (error - self.regularization * self.user_biases[u])
                    self.item_biases[i] += self.learning_rate * (error - self.regularization * self.item_biases[i])
                    
                    # Update factors
                    user_factors_u = self.user_factors[u].copy()
                    item_factors_i = self.item_factors[i].copy()
                    
                    self.user_factors[u] += self.learning_rate * (error * item_factors_i - self.regularization * user_factors_u)
                    self.item_factors[i] += self.learning_rate * (error * user_factors_u - self.regularization * item_factors_i)
            
            # Calculate RMSE on training data
            if (iteration + 1) % 5 == 0 or iteration == 0 or iteration == self.n_iterations - 1:
                rmse = self._calculate_rmse(rating_matrix)
                iter_time = time.time() - iter_start
                logger.info(f"Iteration {iteration+1}/{self.n_iterations}: RMSE = {rmse:.4f}, Time: {iter_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"SGD optimization completed in {total_time:.2f} seconds")
    
    def _calculate_rmse(self, rating_matrix: sp.csr_matrix, predictions: np.ndarray) -> float:
        """
        Calculate RMSE on training data.
        
        Args:
            rating_matrix: User-item rating matrix
            predictions: Predicted ratings
            
        Returns:
            float: Root Mean Squared Error
        """
        # Convert to COO format for easier iteration
        rating_matrix_coo = rating_matrix.tocoo()
        
        n_ratings = len(rating_matrix_coo.data)
        squared_error_sum = 0.0
        
        for u, i, r in zip(rating_matrix_coo.row, rating_matrix_coo.col, rating_matrix_coo.data):
            pred = predictions[u, i]
            squared_error_sum += (r - pred) ** 2
        
        rmse = np.sqrt(squared_error_sum / n_ratings)
        return rmse
    
    def predict(self, user_idx: int, movie_idx: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_idx: User matrix index
            movie_idx: Movie matrix index
            
        Returns:
            float: Predicted rating
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        pred = self.global_bias + self.user_biases[user_idx] + self.item_biases[movie_idx] + \
               np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
        
        # Clip to rating range [1, 5]
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
            # Predict for all items
            # Fast vectorized prediction
            user_bias = self.user_biases[user_idx]
            user_factor = self.user_factors[user_idx]
            
            # Calculate dot product for all items at once
            preds = self.global_bias + user_bias + self.item_biases + np.dot(self.item_factors, user_factor)
            
            # Clip predictions to rating range [1, 5]
            preds = np.clip(preds, 1.0, 5.0)
            
            return preds
        else:
            # Predict for specified items
            predictions = np.zeros(len(movie_indices))
            for i, movie_idx in enumerate(movie_indices):
                predictions[i] = self.predict(user_idx, movie_idx)
            
            return predictions
    
    def recommend_for_user(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_rated: bool = True,
        rating_matrix: Optional[sp.csr_matrix] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            rating_matrix: Optional rating matrix to use for excluding rated items
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get all predictions for this user
        all_predictions = self.predict_for_user(user_idx)
        
        # If exclude_rated and rating_matrix provided, mask out already rated items
        if exclude_rated and rating_matrix is not None:
            user_ratings = rating_matrix[user_idx].toarray().ravel()
            all_predictions[user_ratings > 0] = -np.inf
        
        # Get top-n predictions
        top_indices = np.argsort(all_predictions)[::-1][:n_recommendations]
        
        # Convert to movie IDs and predicted ratings
        recommendations = [
            (self.idx_to_movie[idx], all_predictions[idx])
            for idx in top_indices
        ]
        
        return recommendations

class ALS(MatrixFactorization):
    """Alternating Least Squares matrix factorization."""
    
    def __init__(
        self, 
        n_factors: int = 50,
        regularization: float = 0.1,
        n_iterations: int = 20,
        init_mean: float = 0.0,
        init_std: float = 0.1
    ):
        """
        Initialize ALS matrix factorization model.
        
        Args:
            n_factors: Number of latent factors
            regularization: Regularization parameter
            n_iterations: Number of iterations to run
            init_mean: Mean of initial latent factors
            init_std: Standard deviation of initial latent factors
        """
        super().__init__(
            n_factors=n_factors,
            regularization=regularization,
            n_iterations=n_iterations,
            init_mean=init_mean,
            init_std=init_std,
            method='als'
        )

class SGD(MatrixFactorization):
    """Stochastic Gradient Descent matrix factorization."""
    
    def __init__(
        self, 
        n_factors: int = 50,
        regularization: float = 0.1,
        learning_rate: float = 0.005,
        n_iterations: int = 20,
        init_mean: float = 0.0,
        init_std: float = 0.1
    ):
        """
        Initialize SGD matrix factorization model.
        
        Args:
            n_factors: Number of latent factors
            regularization: Regularization parameter
            learning_rate: Learning rate for SGD
            n_iterations: Number of iterations to run
            init_mean: Mean of initial latent factors
            init_std: Standard deviation of initial latent factors
        """
        super().__init__(
            n_factors=n_factors,
            regularization=regularization,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            init_mean=init_mean,
            init_std=init_std,
            method='sgd'
        )

if __name__ == "__main__":
    # Example usage with MovieLens dataset
    from src.data.data_loader import load_movielens_100k
    from src.features.feature_engineering import create_user_item_matrix, split_train_test_by_time
    
    # Load data
    data = load_movielens_100k()
    
    # Create user-item matrix
    logger.info("Creating user-item matrix")
    rating_matrix, idx_to_user, idx_to_movie, user_to_idx, movie_to_idx = create_user_item_matrix(data["ratings"])
    
    # Create train/test split
    train_ratings, test_ratings = split_train_test_by_time(data["ratings"])
    
    # Create train matrix
    train_matrix, train_idx_to_user, train_idx_to_movie, train_user_to_idx, train_movie_to_idx = create_user_item_matrix(train_ratings)
    
    # Create and fit ALS model
    logger.info("Creating and fitting ALS model")
    als_model = ALS(n_factors=50, regularization=0.1, n_iterations=10)
    als_model.fit(train_matrix, train_idx_to_user, train_idx_to_movie, train_user_to_idx, train_movie_to_idx)
    
    # Generate recommendations for a test user
    test_user_id = 1  # Replace with a user ID from your dataset
    logger.info(f"Generating recommendations for user {test_user_id}")
    recommendations = als_model.recommend_for_user(test_user_id, n_recommendations=5, rating_matrix=train_matrix)
    
    # Display recommendations
    print(f"Top 5 recommendations for user {test_user_id}:")
    for movie_id, predicted_rating in recommendations:
        movie_title = data["movies"][data["movies"]["movie_id"] == movie_id]["title"].values[0]
        print(f"  - {movie_title} (predicted rating: {predicted_rating:.2f})")
    
    # Optional: Create and fit SGD model
    # This is slower so commented out by default
    # logger.info("Creating and fitting SGD model")
    # sgd_model = SGD(n_factors=50, regularization=0.1, learning_rate=0.005, n_iterations=10)
    # sgd_model.fit(train_matrix, train_idx_to_user, train_idx_to_movie, train_user_to_idx, train_movie_to_idx) 