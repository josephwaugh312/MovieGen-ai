"""
Content-Based Filtering module for recommendation systems.
This module provides implementations of content-based recommendation algorithms.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, List, Optional, Union
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GenreBasedRecommender:
    """Recommender based on movie genre similarities."""
    
    def __init__(self):
        """Initialize genre-based recommender."""
        self.movie_genres = None
        self.genre_matrix = None
        self.movie_ids = None
        self.movie_id_to_idx = None
        self.similarity_matrix = None
    
    def fit(self, movie_genres_df: pd.DataFrame):
        """
        Fit the recommender on movie genres data.
        
        Args:
            movie_genres_df: DataFrame with movie_id and genres columns
                             (genres should be a list of genre strings)
        """
        self.movie_genres = movie_genres_df.copy()
        
        # Store movie IDs
        self.movie_ids = self.movie_genres['movie_id'].values
        
        # Create mapping from movie ID to index
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        # Create one-hot encoding for genres
        unique_genres = set()
        for genres in self.movie_genres['genres']:
            unique_genres.update(genres)
        
        # Sort genres for consistent ordering
        genre_list = sorted(unique_genres)
        logger.info(f"Found {len(genre_list)} unique genres")
        
        # Create genre matrix (rows = movies, columns = genres)
        n_movies = len(self.movie_genres)
        n_genres = len(genre_list)
        self.genre_matrix = np.zeros((n_movies, n_genres))
        
        # Fill genre matrix
        for idx, genres in enumerate(self.movie_genres['genres']):
            for genre in genres:
                genre_idx = genre_list.index(genre)
                self.genre_matrix[idx, genre_idx] = 1
        
        # Compute similarity matrix
        logger.info("Computing movie-movie similarity matrix based on genres")
        self.similarity_matrix = np.zeros((n_movies, n_movies))
        
        for i in range(n_movies):
            for j in range(i, n_movies):
                # Use Jaccard similarity for genre vectors
                if np.sum(self.genre_matrix[i]) == 0 or np.sum(self.genre_matrix[j]) == 0:
                    similarity = 0
                else:
                    intersection = np.sum(self.genre_matrix[i] * self.genre_matrix[j])
                    union = np.sum(np.logical_or(self.genre_matrix[i], self.genre_matrix[j]))
                    similarity = intersection / union if union > 0 else 0
                
                self.similarity_matrix[i, j] = similarity
                self.similarity_matrix[j, i] = similarity
        
        logger.info("Genre-based recommender fitted")
        return self
    
    def recommend_similar_movies(
        self, 
        movie_id: int, 
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Recommend movies similar to the given movie based on genres.
        
        Args:
            movie_id: ID of the movie to find similar movies for
            n_recommendations: Number of recommendations to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, similarity) tuples
        """
        if movie_id not in self.movie_id_to_idx:
            logger.warning(f"Movie {movie_id} not found in training data")
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        
        # Get similarities to this movie
        similarities = self.similarity_matrix[movie_idx]
        
        # Filter by minimum similarity and exclude the movie itself
        mask = (similarities >= min_similarity) & (np.arange(len(similarities)) != movie_idx)
        candidate_indices = np.where(mask)[0]
        
        # Sort by similarity (descending)
        sorted_indices = candidate_indices[np.argsort(similarities[candidate_indices])[::-1]]
        
        # Take top-n
        top_indices = sorted_indices[:n_recommendations]
        
        # Convert to movie IDs and similarities
        recommendations = [
            (self.movie_ids[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return recommendations
    
    def recommend_for_user(
        self, 
        user_rated_movies: List[Tuple[int, float]],
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Recommend movies for a user based on their highly-rated movies.
        
        Args:
            user_rated_movies: List of (movie_id, rating) tuples for movies rated by the user
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, score) tuples
        """
        # Filter for positively rated movies (rating >= 4)
        positive_movies = [movie_id for movie_id, rating in user_rated_movies if rating >= 4]
        
        if not positive_movies:
            logger.warning("No positively rated movies found for this user")
            # Fall back to using all rated movies
            positive_movies = [movie_id for movie_id, _ in user_rated_movies]
        
        if not positive_movies:
            logger.warning("No rated movies found for this user")
            return []
        
        # Get recommendations for each positively rated movie
        all_recommendations = []
        for movie_id in positive_movies:
            similar_movies = self.recommend_similar_movies(
                movie_id, 
                n_recommendations=n_recommendations,
                min_similarity=0.1
            )
            all_recommendations.extend(similar_movies)
        
        # Aggregate recommendations
        movie_scores = {}
        for movie_id, similarity in all_recommendations:
            if movie_id in movie_scores:
                movie_scores[movie_id] = max(movie_scores[movie_id], similarity)
            else:
                movie_scores[movie_id] = similarity
        
        # Remove already rated movies if requested
        if exclude_rated:
            rated_movie_ids = [movie_id for movie_id, _ in user_rated_movies]
            for movie_id in rated_movie_ids:
                if movie_id in movie_scores:
                    del movie_scores[movie_id]
        
        # Sort by score
        sorted_recommendations = sorted(
            movie_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        return sorted_recommendations

class TitleBasedRecommender:
    """Recommender based on movie title similarities using TF-IDF."""
    
    def __init__(self):
        """Initialize title-based recommender."""
        self.movies_df = None
        self.title_vectorizer = None
        self.title_matrix = None
        self.movie_ids = None
        self.movie_id_to_idx = None
        self.similarity_matrix = None
    
    def _preprocess_title(self, title: str) -> str:
        """
        Preprocess movie title by extracting year and normalizing.
        
        Args:
            title: Movie title, possibly with year in parentheses
            
        Returns:
            str: Preprocessed title with year appended
        """
        # Extract year if present
        year_match = re.search(r'\((\d{4})\)', title)
        year = year_match.group(1) if year_match else ""
        
        # Remove year and other parenthetical information from title
        clean_title = re.sub(r'\([^)]*\)', '', title).strip()
        
        # Lowercase and normalize
        clean_title = clean_title.lower()
        
        # Return title and year as a single string for vectorization
        return f"{clean_title} {year}".strip()
    
    def fit(self, movies_df: pd.DataFrame):
        """
        Fit the recommender on movie data.
        
        Args:
            movies_df: DataFrame with movie_id and title columns
        """
        self.movies_df = movies_df.copy()
        
        # Store movie IDs
        self.movie_ids = self.movies_df['movie_id'].values
        
        # Create mapping from movie ID to index
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        # Preprocess titles
        preprocessed_titles = [self._preprocess_title(title) for title in self.movies_df['title']]
        
        # Create TF-IDF vectorizer
        self.title_vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Create title matrix
        self.title_matrix = self.title_vectorizer.fit_transform(preprocessed_titles)
        
        # Compute similarity matrix
        logger.info("Computing movie-movie similarity matrix based on titles")
        self.similarity_matrix = cosine_similarity(self.title_matrix)
        
        logger.info("Title-based recommender fitted")
        return self
    
    def recommend_similar_movies(
        self, 
        movie_id: int, 
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Recommend movies similar to the given movie based on title similarity.
        
        Args:
            movie_id: ID of the movie to find similar movies for
            n_recommendations: Number of recommendations to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, similarity) tuples
        """
        if movie_id not in self.movie_id_to_idx:
            logger.warning(f"Movie {movie_id} not found in training data")
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        
        # Get similarities to this movie
        similarities = self.similarity_matrix[movie_idx]
        
        # Filter by minimum similarity and exclude the movie itself
        mask = (similarities >= min_similarity) & (np.arange(len(similarities)) != movie_idx)
        candidate_indices = np.where(mask)[0]
        
        # Sort by similarity (descending)
        sorted_indices = candidate_indices[np.argsort(similarities[candidate_indices])[::-1]]
        
        # Take top-n
        top_indices = sorted_indices[:n_recommendations]
        
        # Convert to movie IDs and similarities
        recommendations = [
            (self.movie_ids[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return recommendations

class HybridContentRecommender:
    """Hybrid recommender combining genre and title-based approaches."""
    
    def __init__(
        self, 
        genre_weight: float = 0.7,
        title_weight: float = 0.3
    ):
        """
        Initialize hybrid content-based recommender.
        
        Args:
            genre_weight: Weight for genre-based recommendations
            title_weight: Weight for title-based recommendations
        """
        self.genre_recommender = GenreBasedRecommender()
        self.title_recommender = TitleBasedRecommender()
        self.genre_weight = genre_weight
        self.title_weight = title_weight
    
    def fit(self, movies_df: pd.DataFrame, movie_genres_df: pd.DataFrame):
        """
        Fit the hybrid recommender.
        
        Args:
            movies_df: DataFrame with movie_id and title columns
            movie_genres_df: DataFrame with movie_id and genres columns
        """
        logger.info("Fitting genre-based recommender")
        self.genre_recommender.fit(movie_genres_df)
        
        logger.info("Fitting title-based recommender")
        self.title_recommender.fit(movies_df)
        
        logger.info("Hybrid content recommender fitted")
        return self
    
    def recommend_similar_movies(
        self, 
        movie_id: int, 
        n_recommendations: int = 10,
        min_similarity: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Recommend movies similar to the given movie.
        
        Args:
            movie_id: ID of the movie to find similar movies for
            n_recommendations: Number of recommendations to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, similarity) tuples
        """
        # Get recommendations from each recommender
        genre_recs = self.genre_recommender.recommend_similar_movies(
            movie_id, 
            n_recommendations=n_recommendations * 2,
            min_similarity=min_similarity
        )
        
        title_recs = self.title_recommender.recommend_similar_movies(
            movie_id, 
            n_recommendations=n_recommendations * 2,
            min_similarity=min_similarity
        )
        
        # Combine recommendations
        movie_scores = {}
        
        for movie_id, score in genre_recs:
            movie_scores[movie_id] = self.genre_weight * score
        
        for movie_id, score in title_recs:
            if movie_id in movie_scores:
                movie_scores[movie_id] += self.title_weight * score
            else:
                movie_scores[movie_id] = self.title_weight * score
        
        # Sort by score
        sorted_recommendations = sorted(
            movie_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        return sorted_recommendations
    
    def recommend_for_user(
        self, 
        user_rated_movies: List[Tuple[int, float]],
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Recommend movies for a user based on their rated movies.
        
        Args:
            user_rated_movies: List of (movie_id, rating) tuples for movies rated by the user
            n_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, score) tuples
        """
        # Get recommendations from genre-based recommender
        genre_recs = self.genre_recommender.recommend_for_user(
            user_rated_movies, 
            n_recommendations=n_recommendations * 2,
            exclude_rated=exclude_rated
        )
        
        # For title-based recommendations, we need to find similar movies for each highly-rated movie
        positive_movies = [movie_id for movie_id, rating in user_rated_movies if rating >= 4]
        
        if not positive_movies:
            # Fall back to using all rated movies
            positive_movies = [movie_id for movie_id, _ in user_rated_movies]
        
        # Get recommendations for each positively rated movie
        title_recs = []
        for movie_id in positive_movies:
            similar_movies = self.title_recommender.recommend_similar_movies(
                movie_id, 
                n_recommendations=n_recommendations,
                min_similarity=0.1
            )
            title_recs.extend(similar_movies)
        
        # Aggregate title-based recommendations
        title_movie_scores = {}
        for movie_id, similarity in title_recs:
            if movie_id in title_movie_scores:
                title_movie_scores[movie_id] = max(title_movie_scores[movie_id], similarity)
            else:
                title_movie_scores[movie_id] = similarity
        
        # Remove already rated movies if requested
        if exclude_rated:
            rated_movie_ids = [movie_id for movie_id, _ in user_rated_movies]
            for movie_id in rated_movie_ids:
                if movie_id in title_movie_scores:
                    del title_movie_scores[movie_id]
        
        # Combine recommendations
        movie_scores = {}
        
        for movie_id, score in genre_recs:
            movie_scores[movie_id] = self.genre_weight * score
        
        for movie_id, score in title_movie_scores.items():
            if movie_id in movie_scores:
                movie_scores[movie_id] += self.title_weight * score
            else:
                movie_scores[movie_id] = self.title_weight * score
        
        # Sort by score
        sorted_recommendations = sorted(
            movie_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        return sorted_recommendations

if __name__ == "__main__":
    # Example usage with MovieLens dataset
    from src.data.data_loader import load_movielens_100k, get_movie_genres
    
    # Load data
    data = load_movielens_100k()
    
    # Create movie genres dataframe
    movie_genres = get_movie_genres(data["movies"], data["genres"])
    
    # Create and fit hybrid recommender
    logger.info("Creating and fitting hybrid content recommender")
    recommender = HybridContentRecommender(genre_weight=0.7, title_weight=0.3)
    recommender.fit(data["movies"], movie_genres)
    
    # Generate recommendations for a specific movie
    test_movie_id = 1  # Toy Story
    logger.info(f"Generating recommendations for movie {test_movie_id}")
    movie_title = data["movies"][data["movies"]["movie_id"] == test_movie_id]["title"].values[0]
    logger.info(f"Selected movie: {movie_title}")
    
    recommendations = recommender.recommend_similar_movies(test_movie_id, n_recommendations=5)
    
    # Display recommendations
    print(f"Top 5 movies similar to '{movie_title}':")
    for movie_id, similarity in recommendations:
        rec_title = data["movies"][data["movies"]["movie_id"] == movie_id]["title"].values[0]
        print(f"  - {rec_title} (similarity: {similarity:.2f})")
    
    # Generate recommendations for a user
    test_user_id = 1
    logger.info(f"Generating recommendations for user {test_user_id}")
    
    # Get movies rated by this user
    user_ratings = data["ratings"][data["ratings"]["user_id"] == test_user_id]
    user_rated_movies = [(row["movie_id"], row["rating"]) for _, row in user_ratings.iterrows()]
    
    user_recommendations = recommender.recommend_for_user(
        user_rated_movies, 
        n_recommendations=5,
        exclude_rated=True
    )
    
    # Display recommendations
    print(f"\nTop 5 recommendations for user {test_user_id}:")
    for movie_id, score in user_recommendations:
        rec_title = data["movies"][data["movies"]["movie_id"] == movie_id]["title"].values[0]
        print(f"  - {rec_title} (score: {score:.2f})") 