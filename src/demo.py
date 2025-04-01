"""
MovieLens Recommender System Demo Application

A simple command-line interface to demonstrate the movie recommendation system.
This demo allows users to:
1. Select a recommendation algorithm
2. Get personalized recommendations
3. See explanations for recommendations
4. Compare different algorithms
"""

import sys
import os
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Any

# Import project modules
from src.data.data_loader import load_movielens_100k, get_movie_genres
from src.features.feature_engineering import create_user_item_matrix, split_train_test_by_time
from src.models.collaborative_filtering import ItemBasedCF, UserBasedCF
from src.models.matrix_factorization import ALS
from src.models.content_based import GenreBasedRecommender, HybridContentRecommender
from src.models.hybrid_recommender import HybridRecommender, get_recommendation_explanation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieLensDemo:
    """Command-line demo for the MovieLens recommender system."""

    def __init__(self):
        """Initialize the demo application."""
        self.data = None
        self.movie_genres = None
        self.recommenders = {}
        self.user_item_matrix = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.loaded = False
        
    def load_data(self):
        """Load the MovieLens dataset and prepare recommenders."""
        if self.loaded:
            return
            
        print("Loading MovieLens dataset...")
        self.data = load_movielens_100k()
        
        # Create movie genres DataFrame
        self.movie_genres = get_movie_genres(self.data["movies"], self.data["genres"])
        
        # Create user-item matrix
        print("Creating user-item matrix...")
        (
            self.user_item_matrix, 
            self.idx_to_user, 
            self.idx_to_movie, 
            self.user_to_idx, 
            self.movie_to_idx
        ) = create_user_item_matrix(self.data["ratings"])
        
        # Load and initialize recommenders
        self._initialize_recommenders()
        
        self.loaded = True
        print("Data loaded and recommenders initialized!")
    
    def _initialize_recommenders(self):
        """Initialize all recommender models."""
        print("Initializing recommender models...")
        
        # Item-based collaborative filtering
        print("  - Building Item-based CF...")
        self.recommenders["itemcf"] = ItemBasedCF(
            similarity_method='cosine',
            min_support=2,
            k_neighbors=30
        ).fit(
            self.user_item_matrix, 
            self.idx_to_user, 
            self.idx_to_movie, 
            self.user_to_idx, 
            self.movie_to_idx
        )
        
        # User-based collaborative filtering
        print("  - Building User-based CF...")
        self.recommenders["usercf"] = UserBasedCF(
            similarity_method='pearson',
            min_support=2,
            k_neighbors=30
        ).fit(
            self.user_item_matrix, 
            self.idx_to_user, 
            self.idx_to_movie, 
            self.user_to_idx, 
            self.movie_to_idx
        )
        
        # Matrix factorization
        print("  - Building ALS...")
        self.recommenders["als"] = ALS(
            n_factors=50,
            regularization=0.1,
            n_iterations=15
        ).fit(
            self.user_item_matrix, 
            self.idx_to_user, 
            self.idx_to_movie, 
            self.user_to_idx, 
            self.movie_to_idx
        )
        
        # Content-based
        print("  - Building Content-based...")
        content_recommender = HybridContentRecommender(
            genre_weight=0.7,
            title_weight=0.3
        )
        content_recommender.fit(self.data["movies"], self.movie_genres)
        self.recommenders["content"] = content_recommender
        
        # Hybrid
        print("  - Building Hybrid...")
        hybrid_recommender = HybridRecommender(
            item_cf_weight=0.4,
            mf_weight=0.3,
            content_weight=0.3,
            n_factors=50,
            item_cf_min_support=2,
            item_cf_k_neighbors=30
        )
        
        hybrid_recommender.fit(
            self.user_item_matrix,
            self.idx_to_user,
            self.idx_to_movie,
            self.user_to_idx,
            self.movie_to_idx,
            self.data["movies"],
            self.movie_genres,
            self.user_item_matrix
        )
        
        self.recommenders["hybrid"] = hybrid_recommender
    
    def get_user_ratings(self, user_id: int) -> List[Tuple[int, str, float]]:
        """
        Get movies rated by a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of (movie_id, movie_title, rating) tuples
        """
        if user_id not in self.user_to_idx:
            return []
            
        user_idx = self.user_to_idx[user_id]
        user_row = self.user_item_matrix[user_idx]
        
        # Extract rated movie indices and ratings
        movie_indices = user_row.indices
        ratings = user_row.data
        
        # Convert to movie IDs, titles, and ratings
        user_ratings = []
        for movie_idx, rating in zip(movie_indices, ratings):
            movie_id = self.idx_to_movie[movie_idx]
            movie_info = self.data["movies"][self.data["movies"]["movie_id"] == movie_id]
            if len(movie_info) > 0:
                movie_title = movie_info.iloc[0]["title"]
                user_ratings.append((movie_id, movie_title, rating))
        
        # Sort by rating (descending)
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        return user_ratings
    
    def print_movie_details(self, movie_id: int):
        """
        Print details for a specific movie.
        
        Args:
            movie_id: Movie ID
        """
        movie_info = self.data["movies"][self.data["movies"]["movie_id"] == movie_id]
        if len(movie_info) == 0:
            print(f"Movie ID {movie_id} not found")
            return
            
        movie = movie_info.iloc[0]
        
        # Get genres
        movie_genres = []
        for _, row in self.movie_genres.iterrows():
            if row["movie_id"] == movie_id:
                movie_genres = row["genres"]
                break
        
        # Calculate average rating
        movie_ratings = self.data["ratings"][self.data["ratings"]["movie_id"] == movie_id]
        avg_rating = movie_ratings["rating"].mean() if len(movie_ratings) > 0 else "No ratings"
        num_ratings = len(movie_ratings)
        
        print("\n" + "=" * 50)
        print(f"Movie: {movie['title']}")
        print("-" * 50)
        print(f"Genres: {', '.join(movie_genres)}")
        print(f"Average Rating: {avg_rating:.2f} ({num_ratings} ratings)")
        print("=" * 50 + "\n")
    
    def get_recommendations(
        self, 
        user_id: int, 
        algorithm: str = "hybrid", 
        n_recommendations: int = 10,
        explain: bool = False
    ):
        """
        Get movie recommendations for a user.
        
        Args:
            user_id: User ID
            algorithm: Recommendation algorithm to use
            n_recommendations: Number of recommendations to generate
            explain: Whether to include explanations
        """
        if not self.loaded:
            self.load_data()
            
        # Check if user exists
        if user_id not in self.user_to_idx:
            print(f"User {user_id} not found in the dataset")
            return
            
        # Get recommender
        algorithm = algorithm.lower()
        if algorithm not in self.recommenders:
            print(f"Algorithm '{algorithm}' not available. Using hybrid recommender.")
            algorithm = "hybrid"
            
        recommender = self.recommenders[algorithm]
        
        # Get recommendations
        print(f"\nGenerating {n_recommendations} recommendations for user {user_id} using {algorithm}...\n")
        
        # Time the recommendation generation
        start_time = time.time()
        recommendations = recommender.recommend_for_user(
            user_id, 
            n_recommendations=n_recommendations,
            exclude_rated=True
        )
        end_time = time.time()
        
        # Print recommendations
        print("-" * 80)
        print(f"Top {n_recommendations} Recommendations for User {user_id}")
        print("-" * 80)
        
        for i, (movie_id, score) in enumerate(recommendations, 1):
            movie_info = self.data["movies"][self.data["movies"]["movie_id"] == movie_id]
            if len(movie_info) > 0:
                movie_title = movie_info.iloc[0]["title"]
                
                # Get genres
                movie_genres = []
                for _, row in self.movie_genres.iterrows():
                    if row["movie_id"] == movie_id:
                        movie_genres = row["genres"]
                        break
                
                print(f"{i}. {movie_title}")
                print(f"   Genres: {', '.join(movie_genres)}")
                print(f"   Score: {score:.4f}")
                
                # Show explanation for hybrid recommender
                if explain and algorithm == "hybrid":
                    explanation = get_recommendation_explanation(
                        user_id, 
                        movie_id, 
                        self.data, 
                        recommender,
                        self.movie_genres
                    )
                    print(f"   Explanation: {explanation}")
                
                print()
        
        print(f"Recommendations generated in {end_time - start_time:.2f} seconds")
        print("-" * 80)
    
    def run_demo(self):
        """Run the interactive demo."""
        if not self.loaded:
            self.load_data()
            
        print("\nWelcome to the MovieLens Recommender System Demo!")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("1. Show user ratings")
            print("2. Get recommendations")
            print("3. Show movie details")
            print("4. Compare algorithms")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                user_id = int(input("Enter user ID: "))
                ratings = self.get_user_ratings(user_id)
                
                if not ratings:
                    print(f"User {user_id} not found or has no ratings")
                    continue
                    
                print(f"\nUser {user_id}'s Ratings:")
                print("-" * 60)
                
                for i, (movie_id, title, rating) in enumerate(ratings[:20], 1):
                    print(f"{i}. {title} - {rating:.1f}/5.0")
                    
                if len(ratings) > 20:
                    print(f"...and {len(ratings) - 20} more")
                
            elif choice == "2":
                user_id = int(input("Enter user ID: "))
                
                print("\nAvailable algorithms:")
                print("1. Item-Based CF (itemcf)")
                print("2. User-Based CF (usercf)")
                print("3. Matrix Factorization (als)")
                print("4. Content-Based (content)")
                print("5. Hybrid (hybrid)")
                
                alg_choice = input("Choose algorithm (1-5, default is hybrid): ")
                
                algorithm_map = {
                    "1": "itemcf",
                    "2": "usercf",
                    "3": "als",
                    "4": "content",
                    "5": "hybrid"
                }
                
                algorithm = algorithm_map.get(alg_choice, "hybrid")
                n_recommendations = int(input("Number of recommendations (default 10): ") or 10)
                explain = input("Show explanations? (y/n, default n): ").lower() == 'y'
                
                self.get_recommendations(user_id, algorithm, n_recommendations, explain)
                
            elif choice == "3":
                movie_id = int(input("Enter movie ID: "))
                self.print_movie_details(movie_id)
                
            elif choice == "4":
                user_id = int(input("Enter user ID: "))
                
                if user_id not in self.user_to_idx:
                    print(f"User {user_id} not found in the dataset")
                    continue
                
                n_recommendations = int(input("Number of recommendations (default 5): ") or 5)
                
                algorithms = ["itemcf", "usercf", "als", "content", "hybrid"]
                results = {}
                
                print("\nComparing recommendation algorithms...")
                
                for algorithm in algorithms:
                    print(f"  - Getting recommendations using {algorithm}...")
                    start_time = time.time()
                    recs = self.recommenders[algorithm].recommend_for_user(
                        user_id, 
                        n_recommendations=n_recommendations,
                        exclude_rated=True
                    )
                    end_time = time.time()
                    
                    # Convert to list of movie titles
                    rec_titles = []
                    for movie_id, score in recs:
                        movie_info = self.data["movies"][self.data["movies"]["movie_id"] == movie_id]
                        if len(movie_info) > 0:
                            rec_titles.append((movie_info.iloc[0]["title"], score))
                    
                    results[algorithm] = {
                        "recommendations": rec_titles,
                        "time": end_time - start_time
                    }
                
                # Print comparison
                print("\nComparison of Recommendations for User", user_id)
                print("=" * 80)
                
                for algorithm, result in results.items():
                    print(f"\n{algorithm.upper()} ({result['time']:.2f} seconds):")
                    print("-" * 60)
                    
                    for i, (title, score) in enumerate(result["recommendations"], 1):
                        print(f"{i}. {title} (score: {score:.4f})")
                
                print("\nNote: The recommendations vary by algorithm due to different approaches:")
                print("- ItemCF: Movies similar to what you've rated highly")
                print("- UserCF: Movies liked by similar users")
                print("- ALS: Uses latent factors from matrix factorization")
                print("- Content: Movies with similar genres/attributes")
                print("- Hybrid: Combines multiple approaches for better recommendations")
                
            elif choice == "5":
                print("\nThank you for using the MovieLens Recommender System Demo!")
                break
                
            else:
                print("Invalid choice, please try again")

def main():
    """Run the MovieLens Recommender System Demo."""
    parser = argparse.ArgumentParser(description='MovieLens Recommender System Demo')
    
    parser.add_argument('--user', type=int, help='User ID to generate recommendations for')
    parser.add_argument('--algorithm', type=str, default='hybrid', 
                        choices=['itemcf', 'usercf', 'als', 'content', 'hybrid'],
                        help='Recommendation algorithm to use')
    parser.add_argument('--recommendations', type=int, default=10,
                        help='Number of recommendations to generate')
    parser.add_argument('--explain', action='store_true',
                        help='Show explanation for recommendations')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive demo mode')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = MovieLensDemo()
    
    if args.interactive:
        # Run interactive demo
        demo.run_demo()
    elif args.user:
        # Generate recommendations for specific user
        demo.get_recommendations(
            args.user, 
            args.algorithm, 
            args.recommendations,
            args.explain
        )
    else:
        # Default to interactive mode if no specific user is provided
        demo.run_demo()

if __name__ == "__main__":
    main() 