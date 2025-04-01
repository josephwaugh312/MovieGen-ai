"""
Hybrid Recommender module combining collaborative filtering and content-based approaches.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Tuple, List, Optional, Union
import logging
import time

# Import our recommender components
from src.models.collaborative_filtering import ItemBasedCF, UserBasedCF
from src.models.matrix_factorization import ALS, SGD
from src.models.content_based import GenreBasedRecommender, TitleBasedRecommender, HybridContentRecommender
from src.models.bert_recommender import BERTContentRecommender, get_bert_recommendation_explanation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridRecommender:
    """
    Hybrid recommender combining multiple approaches:
    - Item-based collaborative filtering
    - Matrix factorization
    - Content-based filtering
    - BERT-based text analysis (optional)
    """
    
    def __init__(
        self,
        item_cf_weight: float = 0.30,
        mf_weight: float = 0.25,
        content_weight: float = 0.20,
        bert_weight: float = 0.25,
        use_bert: bool = True,
        n_factors: int = 50,
        item_cf_min_support: int = 3,
        item_cf_k_neighbors: int = 30,
        bert_model_name: str = "distilbert-base-uncased"
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            item_cf_weight: Weight for item-based collaborative filtering
            mf_weight: Weight for matrix factorization
            content_weight: Weight for content-based recommendations
            bert_weight: Weight for BERT-based recommendations
            use_bert: Whether to use BERT-based recommendations
            n_factors: Number of latent factors for matrix factorization
            item_cf_min_support: Minimum number of common users for item similarity
            item_cf_k_neighbors: Number of neighbors for item-based CF
            bert_model_name: Name of the BERT model to use
        """
        self.item_cf_weight = item_cf_weight
        self.mf_weight = mf_weight
        self.content_weight = content_weight
        self.bert_weight = bert_weight if use_bert else 0.0
        self.use_bert = use_bert
        
        # If not using BERT, redistribute weights
        if not use_bert:
            total = self.item_cf_weight + self.mf_weight + self.content_weight
            self.item_cf_weight = self.item_cf_weight / total
            self.mf_weight = self.mf_weight / total
            self.content_weight = self.content_weight / total
        
        # Initialize component recommenders
        self.item_cf = ItemBasedCF(
            similarity_method='cosine',
            min_support=item_cf_min_support,
            k_neighbors=item_cf_k_neighbors
        )
        
        self.mf = ALS(
            n_factors=n_factors,
            regularization=0.1,
            n_iterations=15
        )
        
        self.content_recommender = HybridContentRecommender(
            genre_weight=0.7,
            title_weight=0.3
        )
        
        # Initialize BERT recommender if requested
        self.bert_recommender = None
        if use_bert:
            self.bert_recommender = BERTContentRecommender(
                bert_model_name=bert_model_name
            )
        
        # Dataset-related attributes
        self.rating_matrix = None
        self.train_matrix = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.user_to_idx = None
        self.movie_to_idx = None
        
        logger.info(f"Initialized HybridRecommender with weights: "
                   f"Item-CF={item_cf_weight}, MF={mf_weight}, "
                   f"Content={content_weight}, BERT={bert_weight}")
    
    def fit(
        self,
        train_matrix: sp.csr_matrix,
        train_idx_to_user: Dict[int, int],
        train_idx_to_movie: Dict[int, int],
        train_user_to_idx: Dict[int, int],
        train_movie_to_idx: Dict[int, int],
        movies_df: pd.DataFrame,
        movie_genres_df: pd.DataFrame,
        full_rating_matrix: Optional[sp.csr_matrix] = None
    ):
        """
        Fit the hybrid recommender.
        
        Args:
            train_matrix: Training user-item matrix
            train_idx_to_user: Mapping from matrix row indices to user IDs (train)
            train_idx_to_movie: Mapping from matrix column indices to movie IDs (train)
            train_user_to_idx: Mapping from user IDs to matrix row indices (train)
            train_movie_to_idx: Mapping from movie IDs to matrix column indices (train)
            movies_df: DataFrame with movie metadata
            movie_genres_df: DataFrame with movie genres
            full_rating_matrix: Full user-item matrix (for evaluation)
        """
        # Store parameters
        self.train_matrix = train_matrix
        self.idx_to_user = train_idx_to_user
        self.idx_to_movie = train_idx_to_movie
        self.user_to_idx = train_user_to_idx
        self.movie_to_idx = train_movie_to_idx
        self.rating_matrix = full_rating_matrix if full_rating_matrix is not None else train_matrix
        
        # Fit item-based CF
        logger.info("Fitting item-based collaborative filtering")
        self.item_cf.fit(
            train_matrix,
            train_idx_to_user,
            train_idx_to_movie,
            train_user_to_idx,
            train_movie_to_idx
        )
        
        # Fit matrix factorization
        logger.info("Fitting matrix factorization (ALS)")
        self.mf.fit(
            train_matrix,
            train_idx_to_user,
            train_idx_to_movie,
            train_user_to_idx,
            train_movie_to_idx
        )
        
        # Fit content-based recommender
        logger.info("Fitting content-based recommender")
        self.content_recommender.fit(movies_df, movie_genres_df)
        
        # Fit BERT recommender if enabled
        if self.use_bert and self.bert_recommender:
            logger.info("Fitting BERT-based recommender")
            
            # Create a description field combining title and genres if needed
            if 'description' not in movies_df.columns:
                movies_df = movies_df.copy()
                movies_df['description'] = movies_df.apply(
                    lambda row: f"{row['title']} is a movie. " + 
                               f"Genres: {', '.join(row['genres'])}" 
                               if isinstance(row.get('genres'), list) else row['title'],
                    axis=1
                )
            
            self.bert_recommender.fit(
                movies_df,
                description_col='description',
                force_recompute=False
            )
        
        logger.info("Hybrid recommender fitted")
        return self
    
    def _get_user_rated_movies(self, user_idx: int) -> List[Tuple[int, float]]:
        """
        Get the movies rated by a user.
        
        Args:
            user_idx: User matrix index
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, rating) tuples
        """
        # Get sparse row for this user
        user_row = self.rating_matrix[user_idx]
        
        # Find non-zero elements (rated movies)
        movie_indices = user_row.indices
        ratings = user_row.data
        
        # Convert to movie IDs
        rated_movies = [
            (self.idx_to_movie[movie_idx], rating)
            for movie_idx, rating in zip(movie_indices, ratings)
        ]
        
        return rated_movies
    
    def recommend_for_user(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_rated: bool = True,
        use_item_cf: bool = True,
        use_mf: bool = True,
        use_content: bool = True,
        use_bert: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            use_item_cf: Whether to use item-based CF
            use_mf: Whether to use matrix factorization
            use_content: Whether to use content-based filtering
            use_bert: Whether to use BERT-based recommendations
            
        Returns:
            List[Tuple[int, float]]: List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Compute number of candidates needed from each recommender
        n_candidates = max(n_recommendations * 3, 50)
        
        # Get user's rated movies for content-based recommendations
        user_rated_movies = self._get_user_rated_movies(user_idx)
        
        # Get recommendations from each component
        item_cf_recs = []
        mf_recs = []
        content_recs = []
        bert_recs = []
        
        if use_item_cf:
            logger.info(f"Getting item-based CF recommendations for user {user_id}")
            item_cf_recs = self.item_cf.recommend_for_user(
                user_id, 
                n_recommendations=n_candidates,
                exclude_rated=exclude_rated
            )
        
        if use_mf:
            logger.info(f"Getting matrix factorization recommendations for user {user_id}")
            mf_recs = self.mf.recommend_for_user(
                user_id, 
                n_recommendations=n_candidates,
                exclude_rated=exclude_rated,
                rating_matrix=self.rating_matrix
            )
        
        if use_content:
            logger.info(f"Getting content-based recommendations for user {user_id}")
            content_recs = self.content_recommender.recommend_for_user(
                user_rated_movies, 
                n_recommendations=n_candidates,
                exclude_rated=exclude_rated
            )
        
        if use_bert and self.use_bert and self.bert_recommender:
            logger.info(f"Getting BERT-based recommendations for user {user_id}")
            bert_recs = self.bert_recommender.recommend_for_user(
                user_rated_movies,
                n_recommendations=n_candidates,
                exclude_rated=exclude_rated
            )
        
        # Combine recommendations with weighted scores
        movie_scores = {}
        
        # Normalize rating predictions to 0-1 scale for better weighting
        max_rating = 5.0
        min_rating = 1.0
        rating_range = max_rating - min_rating
        
        # Add item-based CF recommendations
        for movie_id, rating in item_cf_recs:
            normalized_score = (rating - min_rating) / rating_range
            movie_scores[movie_id] = self.item_cf_weight * normalized_score
        
        # Add matrix factorization recommendations
        for movie_id, rating in mf_recs:
            normalized_score = (rating - min_rating) / rating_range
            if movie_id in movie_scores:
                movie_scores[movie_id] += self.mf_weight * normalized_score
            else:
                movie_scores[movie_id] = self.mf_weight * normalized_score
        
        # Add content-based recommendations
        for movie_id, similarity in content_recs:
            # Content similarity is already normalized
            if movie_id in movie_scores:
                movie_scores[movie_id] += self.content_weight * similarity
            else:
                movie_scores[movie_id] = self.content_weight * similarity
        
        # Add BERT-based recommendations
        for movie_id, similarity in bert_recs:
            # BERT similarity is already normalized
            if movie_id in movie_scores:
                movie_scores[movie_id] += self.bert_weight * similarity
            else:
                movie_scores[movie_id] = self.bert_weight * similarity
        
        # Sort by score
        sorted_recommendations = sorted(
            movie_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        return sorted_recommendations
    
    def explain_recommendation(
        self,
        user_id: int,
        movie_id: int,
        movies_df: pd.DataFrame,
        movie_genres_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Explain why a movie was recommended to a user.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            movies_df: DataFrame with movie metadata
            movie_genres_df: DataFrame with movie genres
            
        Returns:
            Dict: Explanation details
        """
        if user_id not in self.user_to_idx:
            return {"error": f"User {user_id} not found in training data"}
        
        user_idx = self.user_to_idx[user_id]
        
        # Get all movies rated by this user
        user_rated_movies = self._get_user_rated_movies(user_idx)
        
        # Movie title
        movie_title = movies_df[movies_df["movie_id"] == movie_id]["title"].values[0]
        
        # Get movie genres
        movie_genres = []
        for _, row in movie_genres_df.iterrows():
            if row["movie_id"] == movie_id:
                movie_genres = row["genres"]
                break
        
        # Explanation structure
        explanation = {
            "movie_id": movie_id,
            "movie_title": movie_title,
            "user_id": user_id,
            "genres": movie_genres,
            "components": {}
        }
        
        # Item-based CF explanation
        if movie_id in self.movie_to_idx:
            movie_idx = self.movie_to_idx[movie_id]
            
            # Get user's rated movies
            user_ratings = self.rating_matrix[user_idx].toarray().ravel()
            rated_items = np.where(user_ratings > 0)[0]
            
            if len(rated_items) > 0:
                # Get similarities between target movie and rated movies
                item_sims = []
                if movie_idx < self.item_cf.sim_matrix.shape[0]:
                    for item_idx in rated_items:
                        if item_idx < self.item_cf.sim_matrix.shape[1]:
                            sim = self.item_cf.sim_matrix[movie_idx, item_idx]
                            movie_title = movies_df[movies_df["movie_id"] == self.idx_to_movie[item_idx]]["title"].values[0]
                            item_sims.append((self.idx_to_movie[item_idx], movie_title, user_ratings[item_idx], sim))
                
                # Sort by similarity
                item_sims.sort(key=lambda x: x[3], reverse=True)
                
                # Take top 3
                explanation["components"]["item_cf"] = {
                    "similar_movies": item_sims[:3],
                    "explanation": "This movie is similar to movies you've rated highly"
                }
        
        # Content-based explanation
        # Find similar genres in user's highly rated movies
        user_genres = {}
        for movie_id, rating in user_rated_movies:
            if rating >= 4:  # Only consider highly rated movies
                for _, row in movie_genres_df.iterrows():
                    if row["movie_id"] == movie_id:
                        for genre in row["genres"]:
                            if genre in user_genres:
                                user_genres[genre] = max(user_genres[genre], rating)
                            else:
                                user_genres[genre] = rating
        
        # Check which genres of the recommended movie match user preferences
        matching_genres = []
        for genre in movie_genres:
            if genre in user_genres:
                matching_genres.append((genre, user_genres[genre]))
        
        # Sort by rating
        matching_genres.sort(key=lambda x: x[1], reverse=True)
        
        explanation["components"]["content_based"] = {
            "matching_genres": matching_genres,
            "explanation": "This movie has genres you've enjoyed in the past"
        }
        
        # Add BERT-based explanation if available
        if self.use_bert and self.bert_recommender:
            bert_explanation = self.bert_recommender.explain_recommendation(
                movie_id, 
                user_rated_movies
            )
            explanation["components"]["bert"] = bert_explanation
        
        return explanation

def get_recommendation_explanation(
    user_id: int,
    movie_id: int,
    data: Dict[str, pd.DataFrame],
    recommender: HybridRecommender,
    movie_genres_df: pd.DataFrame
) -> str:
    """
    Generate a human-readable explanation for a recommendation.
    
    Args:
        user_id: User ID
        movie_id: Movie ID
        data: Dictionary with MovieLens data DataFrames
        recommender: Fitted HybridRecommender
        movie_genres_df: DataFrame with movie genres
        
    Returns:
        str: Human-readable explanation
    """
    # Get raw explanation data
    explanation = recommender.explain_recommendation(
        user_id, 
        movie_id, 
        data["movies"],
        movie_genres_df
    )
    
    if "error" in explanation:
        return explanation["error"]
    
    movie_title = explanation["movie_title"]
    
    # Build explanation text
    text = f"We recommended '{movie_title}' because:\n\n"
    
    # Add content-based explanation
    if "content_based" in explanation["components"]:
        content_exp = explanation["components"]["content_based"]
        
        if content_exp["matching_genres"]:
            text += "• It has genres you've enjoyed: "
            text += ", ".join([genre for genre, _ in content_exp["matching_genres"][:3]])
            text += "\n"
    
    # Add item-based CF explanation
    if "item_cf" in explanation["components"]:
        item_cf_exp = explanation["components"]["item_cf"]
        
        if "similar_movies" in item_cf_exp and item_cf_exp["similar_movies"]:
            text += "• It's similar to movies you've rated highly:\n"
            
            for _, title, rating, sim in item_cf_exp["similar_movies"][:3]:
                text += f"  - '{title}' (you rated {rating:.1f}/5)\n"
    
    # Add BERT-based explanation
    if "bert" in explanation["components"]:
        bert_exp = explanation["components"]["bert"]
        
        # Add themes explanation
        if bert_exp.get("themes"):
            text += "• It has themes that match your preferences:\n"
            for theme, score in bert_exp["themes"][:2]:
                theme_name = theme.split(":")[0].strip() if ":" in theme else theme
                text += f"  - {theme_name}\n"
        
        # Add sentiment explanation
        if bert_exp.get("sentiment") is not None:
            sentiment_str = "positive" if bert_exp["sentiment"] > 0.6 else "balanced"
            text += f"• It has a {sentiment_str} tone similar to movies you've enjoyed\n"
    
    return text

if __name__ == "__main__":
    # Example usage with MovieLens dataset
    from src.data.data_loader import load_movielens_100k, get_movie_genres
    from src.features.feature_engineering import create_user_item_matrix, split_train_test_by_time
    
    # Load data
    data = load_movielens_100k()
    
    # Create movie genres dataframe
    movie_genres = get_movie_genres(data["movies"], data["genres"])
    
    # Create user-item matrix
    logger.info("Creating user-item matrix")
    rating_matrix, idx_to_user, idx_to_movie, user_to_idx, movie_to_idx = create_user_item_matrix(data["ratings"])
    
    # Create train/test split
    train_ratings, test_ratings = split_train_test_by_time(data["ratings"])
    
    # Create train matrix
    train_matrix, train_idx_to_user, train_idx_to_movie, train_user_to_idx, train_movie_to_idx = create_user_item_matrix(train_ratings)
    
    # Create and fit hybrid recommender
    logger.info("Creating and fitting hybrid recommender")
    recommender = HybridRecommender(
        item_cf_weight=0.4,
        mf_weight=0.3,
        content_weight=0.3,
        bert_weight=0.25,
        use_bert=True,
        n_factors=50,
        item_cf_min_support=3,
        item_cf_k_neighbors=30,
        bert_model_name="distilbert-base-uncased"
    )
    
    recommender.fit(
        train_matrix,
        train_idx_to_user,
        train_idx_to_movie,
        train_user_to_idx,
        train_movie_to_idx,
        data["movies"],
        movie_genres,
        rating_matrix
    )
    
    # Generate recommendations for a test user
    test_user_id = 1  # Replace with a user ID from your dataset
    logger.info(f"Generating recommendations for user {test_user_id}")
    
    recommendations = recommender.recommend_for_user(
        test_user_id, 
        n_recommendations=5,
        exclude_rated=True
    )
    
    # Display recommendations
    print(f"Top 5 recommendations for user {test_user_id}:")
    for movie_id, score in recommendations:
        movie_title = data["movies"][data["movies"]["movie_id"] == movie_id]["title"].values[0]
        print(f"  - {movie_title} (score: {score:.2f})")
        
        # Generate explanation
        explanation = get_recommendation_explanation(
            test_user_id, 
            movie_id, 
            data, 
            recommender,
            movie_genres
        )
        print(f"    {explanation}")
        print() 