"""
Evaluation module for recommender systems.

This module provides functions and classes for evaluating recommendation algorithms
using standard metrics like MAE, RMSE, precision, recall, and diversity measures.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
from collections import defaultdict
import time
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommenderEvaluator:
    """
    A class for evaluating recommender systems using various metrics.
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize evaluator with list of k values for top-k metrics.
        
        Args:
            k_values: List of k values for which to compute top-k metrics
        """
        self.k_values = k_values
        logger.info(f"Initialized RecommenderEvaluator with k values: {k_values}")
    
    def evaluate_rating_prediction(
        self, 
        true_ratings: Union[Dict[Tuple[int, int], float], pd.DataFrame],
        predicted_ratings: Union[Dict[Tuple[int, int], float], pd.DataFrame],
        metrics: List[str] = ['mae', 'rmse']
    ) -> Dict[str, float]:
        """
        Evaluate rating prediction using MAE and RMSE.
        
        Args:
            true_ratings: Dictionary mapping (user_id, movie_id) to true rating,
                or DataFrame with columns user_id, movie_id, rating
            predicted_ratings: Dictionary mapping (user_id, movie_id) to predicted rating,
                or DataFrame with columns user_id, movie_id, rating
            metrics: List of metrics to compute
                
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating rating prediction")
        
        # Convert DataFrames to dictionaries if needed
        if isinstance(true_ratings, pd.DataFrame):
            true_dict = {(row['user_id'], row['movie_id']): row['rating'] 
                        for _, row in true_ratings.iterrows()}
        else:
            true_dict = true_ratings
            
        if isinstance(predicted_ratings, pd.DataFrame):
            pred_dict = {(row['user_id'], row['movie_id']): row['rating'] 
                        for _, row in predicted_ratings.iterrows()}
        else:
            pred_dict = predicted_ratings
        
        # Get common user-item pairs
        common_pairs = set(true_dict.keys()) & set(pred_dict.keys())
        
        if not common_pairs:
            logger.warning("No common user-item pairs found for evaluation")
            return {metric: float('nan') for metric in metrics}
        
        # Extract true and predicted ratings
        y_true = [true_dict[pair] for pair in common_pairs]
        y_pred = [pred_dict[pair] for pair in common_pairs]
        
        # Compute metrics
        results = {}
        
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_true, y_pred)
            
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
        logger.info(f"Rating prediction results: {results}")
        return results
    
    def _precision_recall_at_k(
        self,
        true_items: Dict[int, List[int]],
        recommended_items: Dict[int, List[int]],
        k: int
    ) -> Tuple[float, float]:
        """
        Calculate precision and recall at k.
        
        Args:
            true_items: Dictionary mapping user_id to list of relevant item_ids
            recommended_items: Dictionary mapping user_id to list of recommended item_ids
            k: Number of recommendations to consider
            
        Returns:
            Tuple with (precision@k, recall@k)
        """
        precisions = []
        recalls = []
        
        for user_id, true_items_user in true_items.items():
            if user_id not in recommended_items:
                continue
                
            # Get top-k recommendations
            recs = recommended_items[user_id][:k]
            
            # Calculate metrics
            n_relevant = len(set(true_items_user) & set(recs))
            
            precision = n_relevant / min(k, len(recs)) if recs else 0
            recall = n_relevant / len(true_items_user) if true_items_user else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Average over all users
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        
        return avg_precision, avg_recall
    
    def evaluate_ranking(
        self,
        true_items: Dict[int, List[int]],
        recommended_items: Dict[int, List[int]],
        all_items: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate recommendation ranking using precision, recall, and other metrics.
        
        Args:
            true_items: Dictionary mapping user_id to list of relevant item_ids
            recommended_items: Dictionary mapping user_id to list of recommended item_ids
            all_items: List of all possible item_ids (for diversity calculation)
            
        Returns:
            Dictionary with evaluation results for each k value
        """
        logger.info("Evaluating recommendation ranking")
        
        results = {}
        
        # Compute precision and recall at each k
        for k in self.k_values:
            precision, recall = self._precision_recall_at_k(true_items, recommended_items, k)
            
            results[f'precision@{k}'] = precision
            results[f'recall@{k}'] = recall
            
            # F1 score
            if precision + recall > 0:
                results[f'f1@{k}'] = 2 * precision * recall / (precision + recall)
            else:
                results[f'f1@{k}'] = 0
        
        # Calculate diversity if all items are provided
        if all_items:
            results['diversity'] = self._calculate_diversity(recommended_items, all_items)
        
        logger.info(f"Ranking evaluation results: {results}")
        return results
    
    def _calculate_diversity(
        self,
        recommended_items: Dict[int, List[int]],
        all_items: List[int],
        max_k: Optional[int] = None
    ) -> float:
        """
        Calculate diversity of recommendations as unique ratio.
        
        Args:
            recommended_items: Dictionary mapping user_id to list of recommended item_ids
            all_items: List of all possible item_ids
            max_k: Maximum number of recommendations to consider
            
        Returns:
            Diversity score (0-1)
        """
        if not recommended_items:
            return 0
            
        # Set maximum k if not specified
        if max_k is None:
            max_k = max(self.k_values)
        
        # Count unique items across all recommendations
        unique_items = set()
        for user_id, items in recommended_items.items():
            unique_items.update(items[:max_k])
        
        # Calculate diversity as ratio of unique items to all recommended items
        diversity = len(unique_items) / len(all_items) if all_items else 0
        
        return diversity

    def evaluate_recommender(
        self,
        recommender: Any,
        test_data: Union[pd.DataFrame, Dict[int, List[Tuple[int, float]]]],
        all_movies: Optional[List[int]] = None,
        n_recommendations: int = 20,
        include_rating_metrics: bool = True,
        include_ranking_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a recommender model on test data.
        
        Args:
            recommender: Recommender model with recommend_for_user method
            test_data: Test data with user ratings
                (DataFrame with user_id, movie_id, rating columns or
                Dictionary mapping user_id to list of (movie_id, rating) tuples)
            all_movies: List of all movie IDs (for diversity calculation)
            n_recommendations: Number of recommendations to generate
            include_rating_metrics: Whether to include rating prediction metrics
            include_ranking_metrics: Whether to include ranking metrics
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating recommender with {n_recommendations} recommendations")
        
        # Start timer
        start_time = time.time()
        
        # Convert test data to dictionary if DataFrame
        if isinstance(test_data, pd.DataFrame):
            user_ratings = defaultdict(list)
            for _, row in test_data.iterrows():
                user_ratings[row['user_id']].append((row['movie_id'], row['rating']))
        else:
            user_ratings = test_data
        
        # Get all users in test set
        test_users = list(user_ratings.keys())
        logger.info(f"Evaluating on {len(test_users)} test users")
        
        # For rating metrics
        true_ratings = {}
        pred_ratings = {}
        
        # For ranking metrics
        true_relevant_items = {}  # user_id -> [movie_id, ...]
        recommended_items = {}    # user_id -> [movie_id, ...]
        
        # Define threshold for relevant items (e.g., rating >= 4)
        relevance_threshold = 4.0
        
        # Generate recommendations for each user
        for user_id in test_users:
            # Get true ratings for this user
            true_user_ratings = user_ratings[user_id]
            
            # Get high-rated items as relevant items for ranking metrics
            true_relevant_items[user_id] = [
                movie_id for movie_id, rating in true_user_ratings
                if rating >= relevance_threshold
            ]
            
            # Skip users with no relevant items
            if include_ranking_metrics and not true_relevant_items[user_id]:
                continue
            
            # Get movie IDs and ratings that this user rated
            rated_movie_ids = [movie_id for movie_id, _ in true_user_ratings]
            
            try:
                # Get recommendations for this user
                user_recommendations = recommender.recommend_for_user(
                    user_id,
                    n_recommendations=n_recommendations,
                    exclude_rated=True
                )
                
                # Store recommended items for ranking metrics
                recommended_items[user_id] = [movie_id for movie_id, _ in user_recommendations]
                
                # For rating metrics, try to predict ratings for items in test set
                if include_rating_metrics and hasattr(recommender, 'predict_rating'):
                    for movie_id, true_rating in true_user_ratings:
                        true_ratings[(user_id, movie_id)] = true_rating
                        
                        try:
                            # Predict rating for this user-movie pair
                            pred_rating = recommender.predict_rating(user_id, movie_id)
                            pred_ratings[(user_id, movie_id)] = pred_rating
                        except Exception as e:
                            logger.warning(f"Error predicting rating for user {user_id}, movie {movie_id}: {e}")
            
            except Exception as e:
                logger.warning(f"Error generating recommendations for user {user_id}: {e}")
        
        # Compile results
        results = {}
        
        # Rating prediction metrics
        if include_rating_metrics and pred_ratings:
            rating_metrics = self.evaluate_rating_prediction(true_ratings, pred_ratings)
            results.update(rating_metrics)
        
        # Ranking metrics
        if include_ranking_metrics and recommended_items:
            ranking_metrics = self.evaluate_ranking(
                true_relevant_items, 
                recommended_items,
                all_movies
            )
            results.update(ranking_metrics)
        
        # Add execution time
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time
        
        logger.info(f"Evaluation completed in {execution_time:.2f} seconds")
        return results

def create_test_items_for_users(
    ratings_df: pd.DataFrame,
    train_ratio: float = 0.8,
    rating_threshold: float = 4.0,
    min_ratings: int = 5,
    random_state: int = 42
) -> Tuple[Dict[int, List[int]], Dict[int, List[Tuple[int, float]]]]:
    """
    Split ratings into training and testing sets, and create test relevant items.
    
    Args:
        ratings_df: DataFrame with user_id, movie_id, rating columns
        train_ratio: Ratio of data to use for training
        rating_threshold: Threshold to consider an item relevant
        min_ratings: Minimum number of ratings per user
        random_state: Random seed
        
    Returns:
        Tuple with (
            Dictionary mapping user_id to list of relevant test item_ids,
            Dictionary mapping user_id to list of (movie_id, rating) tuples for testing
        )
    """
    np.random.seed(random_state)
    
    # Group by user
    user_groups = ratings_df.groupby('user_id')
    
    test_relevant_items = {}  # For ranking evaluation
    test_user_ratings = {}    # For rating prediction evaluation
    
    for user_id, user_data in user_groups:
        # Skip users with too few ratings
        if len(user_data) < min_ratings:
            continue
        
        # Split randomly
        n_train = int(len(user_data) * train_ratio)
        indices = np.random.permutation(len(user_data))
        test_indices = indices[n_train:]
        
        # Extract test data
        user_test_data = user_data.iloc[test_indices]
        
        # Get relevant items (high ratings)
        relevant_items = user_test_data[user_test_data['rating'] >= rating_threshold]['movie_id'].tolist()
        
        # Only include users with at least one relevant test item
        if relevant_items:
            test_relevant_items[user_id] = relevant_items
            test_user_ratings[user_id] = [
                (row['movie_id'], row['rating']) 
                for _, row in user_test_data.iterrows()
            ]
    
    logger.info(f"Created test sets for {len(test_relevant_items)} users")
    return test_relevant_items, test_user_ratings

def compare_recommenders(
    recommenders: Dict[str, Any],
    test_data: Union[pd.DataFrame, Dict[int, List[Tuple[int, float]]]],
    all_movies: Optional[List[int]] = None,
    metrics: List[str] = ['rmse', 'precision@10', 'recall@10', 'diversity'],
    n_recommendations: int = 20
) -> pd.DataFrame:
    """
    Compare multiple recommender systems on the same test data.
    
    Args:
        recommenders: Dictionary mapping recommender name to recommender object
        test_data: Test data with user ratings
        all_movies: List of all movie IDs
        metrics: List of metrics to include in comparison
        n_recommendations: Number of recommendations to generate
        
    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing {len(recommenders)} recommender systems")
    
    # Initialize evaluator
    max_k = max([int(m.split('@')[1]) for m in metrics if '@' in m] + [10])
    evaluator = RecommenderEvaluator(k_values=[5, 10, max_k])
    
    # Store results
    results = {}
    
    # Evaluate each recommender
    for name, recommender in recommenders.items():
        logger.info(f"Evaluating recommender: {name}")
        
        try:
            # Evaluate recommender
            eval_results = evaluator.evaluate_recommender(
                recommender,
                test_data,
                all_movies,
                n_recommendations=n_recommendations,
                include_rating_metrics='rmse' in metrics or 'mae' in metrics,
                include_ranking_metrics=any('@' in m for m in metrics)
            )
            
            # Filter to requested metrics
            filtered_results = {m: eval_results.get(m, float('nan')) for m in metrics}
            results[name] = filtered_results
            
        except Exception as e:
            logger.error(f"Error evaluating recommender {name}: {e}")
            results[name] = {m: float('nan') for m in metrics}
    
    # Convert to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Add execution time if available
    if 'execution_time' in eval_results:
        results_df['execution_time'] = [results[name].get('execution_time', float('nan')) 
                                        for name in results]
    
    return results_df

if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import load_movielens_100k
    from src.features.feature_engineering import create_user_item_matrix, split_train_test_by_time
    from src.models.collaborative_filtering import ItemBasedCF
    from src.models.matrix_factorization import ALS
    from src.models.content_based import HybridContentRecommender
    from src.models.hybrid_recommender import HybridRecommender
    
    # Load data
    data = load_movielens_100k()
    
    # Split into train and test
    train_ratings, test_ratings = split_train_test_by_time(data["ratings"])
    
    # Create user-item matrix for training
    train_matrix, train_idx_to_user, train_idx_to_movie, train_user_to_idx, train_movie_to_idx = create_user_item_matrix(train_ratings)
    
    # Build and fit recommenders
    recommenders = {
        "ItemCF": ItemBasedCF(min_support=3, k_neighbors=30).fit(
            train_matrix, train_idx_to_user, train_idx_to_movie, train_user_to_idx, train_movie_to_idx
        ),
        "ALS": ALS(n_factors=50).fit(
            train_matrix, train_idx_to_user, train_idx_to_movie, train_user_to_idx, train_movie_to_idx
        )
    }
    
    # Create test items
    test_relevant_items, test_user_ratings = create_test_items_for_users(
        test_ratings, train_ratio=0, rating_threshold=4.0
    )
    
    # Get all movie IDs
    all_movies = data["movies"]["movie_id"].tolist()
    
    # Compare recommenders
    results = compare_recommenders(
        recommenders,
        test_user_ratings,
        all_movies,
        metrics=['rmse', 'precision@5', 'precision@10', 'recall@10', 'diversity']
    )
    
    print("\nRecommender System Comparison:")
    print(results) 