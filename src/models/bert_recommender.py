"""
BERT-based Recommender module using transformer embeddings.

This module implements a recommendation approach that uses BERT embeddings
of movie descriptions to find similar movies and generate recommendations.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity

from src.models.text_analysis import MovieTextAnalyzer, embed_movie_descriptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BERTContentRecommender:
    """
    Recommender using BERT embeddings of movie descriptions.
    
    This recommender uses transformer models to generate dense vector embeddings
    for movie descriptions and finds similar movies based on these embeddings.
    """
    
    def __init__(
        self,
        bert_model_name: str = "distilbert-base-uncased",
        use_sentiment: bool = True,
        use_themes: bool = True,
        embedding_weight: float = 0.7,
        sentiment_weight: float = 0.1,
        theme_weight: float = 0.2,
        embedding_cache_file: Optional[str] = "data/processed/movie_embeddings.npy",
        analysis_cache_file: Optional[str] = "data/processed/text_analysis_results.pkl"
    ):
        """
        Initialize the BERT-based recommender.
        
        Args:
            bert_model_name: Name of the BERT model to use
            use_sentiment: Whether to use sentiment analysis
            use_themes: Whether to use theme extraction
            embedding_weight: Weight for embedding similarity
            sentiment_weight: Weight for sentiment similarity
            theme_weight: Weight for theme similarity
            embedding_cache_file: File to cache embeddings
            analysis_cache_file: File to cache text analysis results
        """
        self.bert_model_name = bert_model_name
        self.use_sentiment = use_sentiment
        self.use_themes = use_themes
        self.embedding_weight = embedding_weight
        self.sentiment_weight = sentiment_weight
        self.theme_weight = theme_weight
        
        self.embedding_cache_file = embedding_cache_file
        self.analysis_cache_file = analysis_cache_file
        
        # Initialize attributes
        self.analyzer = None
        self.movie_embeddings = None
        self.movie_sentiments = None
        self.movie_theme_scores = None
        self.theme_descriptions = None
        self.movie_to_idx = None
        self.movie_ids = None
        self.similarity_matrix = None
        self.movies_df = None
        
        logger.info(f"Initialized BERTContentRecommender with {bert_model_name}")
    
    def _load_or_create_analyzer(self):
        """Load or create the text analyzer."""
        if self.analyzer is None:
            self.analyzer = MovieTextAnalyzer(model_name=self.bert_model_name)
        return self.analyzer
    
    def fit(
        self, 
        movies_df: pd.DataFrame, 
        description_col: str = 'description',
        force_recompute: bool = False
    ):
        """
        Fit the recommender with movie data.
        
        Args:
            movies_df: DataFrame with movie data
            description_col: Column name for movie descriptions
            force_recompute: Whether to force recomputation of embeddings
            
        Returns:
            self
        """
        self.movies_df = movies_df.copy()
        self.movie_ids = movies_df['movie_id'].values
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        # Try to load cached results
        loaded_from_cache = False
        if not force_recompute and self.analysis_cache_file and os.path.exists(self.analysis_cache_file):
            try:
                logger.info(f"Loading text analysis results from {self.analysis_cache_file}")
                with open(self.analysis_cache_file, 'rb') as f:
                    analysis_results = pickle.load(f)
                    
                self.movie_embeddings = analysis_results.get('embeddings')
                self.movie_sentiments = analysis_results.get('positive_sentiment')
                self.movie_theme_scores = analysis_results.get('theme_scores')
                self.theme_descriptions = analysis_results.get('theme_descriptions')
                
                # Verify shapes match
                if self.movie_embeddings is not None and len(self.movie_embeddings) == len(movies_df):
                    loaded_from_cache = True
                    logger.info(f"Loaded embeddings with shape {self.movie_embeddings.shape}")
                else:
                    logger.warning("Cached embeddings don't match movies_df, recomputing")
            except Exception as e:
                logger.warning(f"Error loading cached results: {e}, recomputing")
        
        # Compute embeddings if not loaded from cache
        if not loaded_from_cache:
            logger.info(f"Computing text analysis for {len(movies_df)} movies")
            analyzer = self._load_or_create_analyzer()
            
            analysis_results = analyzer.process_movie_dataframe(
                movies_df,
                description_col=description_col,
                save_to_file=self.analysis_cache_file if self.analysis_cache_file else None
            )
            
            self.movie_embeddings = analysis_results['embeddings']
            self.movie_sentiments = analysis_results['positive_sentiment']
            self.movie_theme_scores = analysis_results['theme_scores']
            self.theme_descriptions = analysis_results['theme_descriptions']
        
        # Compute similarity matrix
        self._compute_similarity_matrix()
        
        return self
    
    def _compute_similarity_matrix(self):
        """Compute the similarity matrix between movies."""
        logger.info("Computing movie similarity matrix")
        
        # Initialize with embedding similarity
        embedding_sim = cosine_similarity(self.movie_embeddings)
        
        # Start with embedding similarity (base component)
        similarity = self.embedding_weight * embedding_sim
        
        # Add sentiment similarity if available and enabled
        if self.use_sentiment and self.movie_sentiments is not None:
            sentiment_array = np.array(self.movie_sentiments).reshape(-1, 1)
            sentiment_sim = 1 - np.abs(sentiment_array - sentiment_array.T)
            similarity += self.sentiment_weight * sentiment_sim
        
        # Add theme similarity if available and enabled
        if self.use_themes and self.movie_theme_scores is not None and self.movie_theme_scores.shape[1] > 0:
            theme_sim = cosine_similarity(self.movie_theme_scores)
            similarity += self.theme_weight * theme_sim
        
        # Normalize and store
        self.similarity_matrix = similarity
        
        logger.info(f"Computed similarity matrix with shape {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def get_similar_movies(
        self, 
        movie_id: int, 
        n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get movies similar to the given movie.
        
        Args:
            movie_id: Movie ID to find similar movies for
            n_recommendations: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if movie_id not in self.movie_to_idx:
            logger.warning(f"Movie ID {movie_id} not found in the dataset")
            return []
        
        movie_idx = self.movie_to_idx[movie_id]
        
        # Get similarities to this movie
        similarities = self.similarity_matrix[movie_idx]
        
        # Get top similar movies (excluding self)
        similar_indices = similarities.argsort()[::-1]
        similar_indices = similar_indices[similar_indices != movie_idx][:n_recommendations]
        
        # Convert to movie IDs and scores
        similar_movies = [
            (self.movie_ids[idx], similarities[idx])
            for idx in similar_indices
        ]
        
        return similar_movies
    
    def get_movie_themes(self, movie_id: int) -> List[Tuple[str, float]]:
        """
        Get themes for a specific movie with scores.
        
        Args:
            movie_id: Movie ID
            
        Returns:
            List of (theme_description, score) tuples
        """
        if movie_id not in self.movie_to_idx or self.theme_descriptions is None:
            return []
        
        movie_idx = self.movie_to_idx[movie_id]
        
        # Get theme scores for this movie
        if self.movie_theme_scores is not None and movie_idx < len(self.movie_theme_scores):
            theme_scores = self.movie_theme_scores[movie_idx]
            
            # Match with descriptions
            movie_themes = [
                (desc, score) 
                for desc, score in zip(self.theme_descriptions, theme_scores)
                if score > 0.1  # Only include relevant themes
            ]
            
            # Sort by score
            movie_themes.sort(key=lambda x: x[1], reverse=True)
            return movie_themes
        
        return []
    
    def recommend_for_user(
        self,
        user_rated_movies: List[Tuple[int, float]],
        n_recommendations: int = 10,
        exclude_rated: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user based on their rated movies.
        
        Args:
            user_rated_movies: List of (movie_id, rating) tuples
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        # Get rated movie IDs
        rated_movie_ids = [movie_id for movie_id, _ in user_rated_movies]
        
        # Get highly rated movies (rating >= 4)
        liked_movies = [(movie_id, rating) for movie_id, rating in user_rated_movies if rating >= 4]
        
        if not liked_movies:
            logger.warning("No highly rated movies found for user, using all rated movies")
            liked_movies = user_rated_movies
        
        # Calculate a weighted average of similar movies
        movie_scores = {}
        
        for movie_id, rating in liked_movies:
            # Get similar movies to this movie
            similar_movies = self.get_similar_movies(movie_id, n_recommendations * 2)
            
            # Add to scores, weighted by user rating
            for similar_id, similarity in similar_movies:
                # Skip if already rated and exclude_rated is True
                if exclude_rated and similar_id in rated_movie_ids:
                    continue
                    
                # Normalize rating to 0-1 scale and use as weight
                rating_weight = (rating - 1) / 4  # Assuming 1-5 scale
                weighted_score = similarity * rating_weight
                
                if similar_id in movie_scores:
                    movie_scores[similar_id] = max(movie_scores[similar_id], weighted_score)
                else:
                    movie_scores[similar_id] = weighted_score
        
        # Sort by score
        recommendations = sorted(
            movie_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        return recommendations
    
    def explain_recommendation(
        self, 
        movie_id: int, 
        user_rated_movies: List[Tuple[int, float]]
    ) -> Dict[str, Any]:
        """
        Explain why a movie was recommended to a user.
        
        Args:
            movie_id: Movie ID that was recommended
            user_rated_movies: List of (movie_id, rating) tuples for the user
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            "movie_id": movie_id,
            "similar_movies": [],
            "themes": [],
            "sentiment": None
        }
        
        if movie_id not in self.movie_to_idx:
            return explanation
        
        movie_idx = self.movie_to_idx[movie_id]
        
        # Get movie title
        if self.movies_df is not None:
            movie_info = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if len(movie_info) > 0:
                explanation["movie_title"] = movie_info.iloc[0]["title"]
        
        # Find which user-rated movies are most similar to this movie
        similar_to_rated = []
        
        for rated_id, rating in user_rated_movies:
            if rated_id not in self.movie_to_idx:
                continue
                
            rated_idx = self.movie_to_idx[rated_id]
            similarity = self.similarity_matrix[movie_idx, rated_idx]
            
            # Only include if reasonably similar
            if similarity > 0.3:
                # Get title of rated movie
                rated_title = ""
                if self.movies_df is not None:
                    rated_info = self.movies_df[self.movies_df["movie_id"] == rated_id]
                    if len(rated_info) > 0:
                        rated_title = rated_info.iloc[0]["title"]
                
                similar_to_rated.append({
                    "movie_id": rated_id,
                    "title": rated_title,
                    "rating": rating,
                    "similarity": similarity
                })
        
        # Sort by similarity * rating
        similar_to_rated.sort(
            key=lambda x: x["similarity"] * x["rating"], 
            reverse=True
        )
        explanation["similar_movies"] = similar_to_rated[:3]  # Top 3
        
        # Add themes
        explanation["themes"] = self.get_movie_themes(movie_id)
        
        # Add sentiment
        if self.movie_sentiments is not None and movie_idx < len(self.movie_sentiments):
            explanation["sentiment"] = self.movie_sentiments[movie_idx]
        
        return explanation

def get_bert_recommendation_explanation(
    movie_id: int,
    user_rated_movies: List[Tuple[int, float]],
    recommender: BERTContentRecommender,
    movies_df: pd.DataFrame
) -> str:
    """
    Generate a human-readable explanation for a BERT-based recommendation.
    
    Args:
        movie_id: Movie ID that was recommended
        user_rated_movies: List of (movie_id, rating) tuples for the user
        recommender: Fitted BERTContentRecommender
        movies_df: DataFrame with movie data
        
    Returns:
        str: Human-readable explanation
    """
    explanation = recommender.explain_recommendation(movie_id, user_rated_movies)
    
    # Get movie title
    movie_title = explanation.get("movie_title", f"Movie {movie_id}")
    
    text = f"We recommended '{movie_title}' because:\n\n"
    
    # Add similar movies explanation
    if explanation["similar_movies"]:
        text += "• It's similar to movies you've rated highly:\n"
        for movie in explanation["similar_movies"]:
            title = movie.get("title", f"Movie {movie['movie_id']}")
            text += f"  - '{title}' (you rated {movie['rating']:.1f}/5)\n"
        text += "\n"
    
    # Add themes explanation
    if explanation["themes"]:
        text += "• It has these themes that match your preferences:\n"
        for theme, score in explanation["themes"][:3]:
            theme_name = theme.split(":")[0].strip()
            text += f"  - {theme_name} ({score:.2f})\n"
        text += "\n"
    
    # Add sentiment explanation
    if explanation["sentiment"] is not None:
        sentiment_str = "positive" if explanation["sentiment"] > 0.6 else "balanced"
        text += f"• It has a {sentiment_str} tone that matches your preferred movies\n"
    
    return text

if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import load_movielens_100k, get_movie_genres
    
    # Load data
    data = load_movielens_100k()
    movies_df = data["movies"]
    
    # Create movie descriptions (using title and genres)
    movie_genres = get_movie_genres(movies_df, data["genres"])
    
    movies_df["description"] = movies_df.apply(
        lambda row: f"{row['title']} is a {' '.join(row['genres'])} movie." 
        if row["movie_id"] in movie_genres.index 
        else row["title"],
        axis=1
    )
    
    # Create and fit recommender
    recommender = BERTContentRecommender()
    recommender.fit(movies_df, description_col="description")
    
    # Get similar movies example
    movie_id = 1  # Example movie ID
    similar_movies = recommender.get_similar_movies(movie_id, 5)
    
    print(f"Movies similar to {movies_df[movies_df['movie_id'] == movie_id].iloc[0]['title']}:")
    for similar_id, score in similar_movies:
        similar_title = movies_df[movies_df['movie_id'] == similar_id].iloc[0]['title']
        print(f"- {similar_title} (similarity: {score:.4f})") 