"""
BERT-based Movie Recommendation Demo

This script demonstrates the BERT-based recommender system for movies.
It downloads transformers models, analyzes movie descriptions using BERT,
and generates recommendations based on semantic understanding of movie content.
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
from src.features.feature_engineering import create_user_item_matrix
from src.models.bert_recommender import BERTContentRecommender
from src.models.hybrid_recommender import HybridRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BERTMovieLensDemo:
    """Demo for the BERT-based movie recommendation system."""
    
    def __init__(self, bert_model_name: str = "distilbert-base-uncased"):
        """
        Initialize the demo.
        
        Args:
            bert_model_name: Name of the BERT model to use
        """
        self.bert_model_name = bert_model_name
        self.data = None
        self.movie_genres = None
        self.movies_df = None
        self.bert_recommender = None
        self.hybrid_recommender = None
        self.user_item_matrix = None
        self.idx_to_user = None
        self.idx_to_movie = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.loaded = False
    
    def load_data(self):
        """Load data and initialize recommenders."""
        if self.loaded:
            return
        
        print("Loading MovieLens dataset...")
        self.data = load_movielens_100k()
        
        # Prepare movie data
        self.movies_df = self.data["movies"]
        self.movie_genres = get_movie_genres(self.movies_df, self.data["genres"])
        
        # Create movie descriptions by combining title and genres
        self.movies_df["description"] = self.movies_df.apply(
            lambda row: f"{row['title']} is a " + 
                       f"{', '.join(self.movie_genres.loc[row['movie_id'], 'genres'] if row['movie_id'] in self.movie_genres.index else [])} movie.",
            axis=1
        )
        
        # Create user-item matrix
        print("Creating user-item matrix...")
        (
            self.user_item_matrix, 
            self.idx_to_user, 
            self.idx_to_movie, 
            self.user_to_idx, 
            self.movie_to_idx
        ) = create_user_item_matrix(self.data["ratings"])
        
        # Initialize BERT recommender
        print(f"Initializing BERT recommender with {self.bert_model_name}...")
        self.bert_recommender = BERTContentRecommender(
            bert_model_name=self.bert_model_name
        )
        
        # Initialize hybrid recommender
        print("Initializing hybrid recommender with BERT component...")
        self.hybrid_recommender = HybridRecommender(
            item_cf_weight=0.25,
            mf_weight=0.25,
            content_weight=0.20,
            bert_weight=0.30,
            use_bert=True,
            bert_model_name=self.bert_model_name
        )
        
        self.loaded = True
    
    def build_recommenders(self):
        """Build and fit the recommenders."""
        if not self.loaded:
            self.load_data()
        
        # Fit BERT recommender
        print("Fitting BERT recommender (this may take a while the first time)...")
        start_time = time.time()
        self.bert_recommender.fit(self.movies_df, description_col="description")
        bert_time = time.time() - start_time
        print(f"BERT recommender fitted in {bert_time:.2f} seconds")
        
        # Fit hybrid recommender
        print("Fitting hybrid recommender...")
        start_time = time.time()
        self.hybrid_recommender.fit(
            self.user_item_matrix,
            self.idx_to_user,
            self.idx_to_movie,
            self.user_to_idx,
            self.movie_to_idx,
            self.movies_df,
            self.movie_genres,
            self.user_item_matrix
        )
        hybrid_time = time.time() - start_time
        print(f"Hybrid recommender fitted in {hybrid_time:.2f} seconds")
    
    def get_user_ratings(self, user_id: int) -> List[Tuple[int, str, float]]:
        """
        Get movies rated by a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of (movie_id, movie_title, rating) tuples
        """
        if not self.loaded:
            self.load_data()
            
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
            movie_info = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if len(movie_info) > 0:
                movie_title = movie_info.iloc[0]["title"]
                user_ratings.append((movie_id, movie_title, rating))
        
        # Sort by rating (descending)
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        return user_ratings
    
    def analyze_movie(self, movie_id: int):
        """
        Analyze a movie using BERT and display themes and sentiment.
        
        Args:
            movie_id: Movie ID to analyze
        """
        if not self.loaded or self.bert_recommender is None:
            print("Loading data and BERT recommender...")
            self.load_data()
            self.build_recommenders()
        
        # Get movie info
        movie_info = self.movies_df[self.movies_df["movie_id"] == movie_id]
        if len(movie_info) == 0:
            print(f"Movie ID {movie_id} not found")
            return
            
        movie = movie_info.iloc[0]
        
        # Get movie description
        description = movie["description"] if "description" in movie else movie["title"]
        
        print("\n" + "=" * 60)
        print(f"BERT Analysis of Movie: {movie['title']}")
        print("-" * 60)
        
        # Get movie themes
        themes = self.bert_recommender.get_movie_themes(movie_id)
        
        print("Description:")
        print(f"  {description}")
        print("\nThemes detected:")
        if themes:
            for theme, score in themes:
                print(f"  - {theme} (score: {score:.2f})")
        else:
            print("  No significant themes detected")
        
        # Get similar movies
        similar_movies = self.bert_recommender.get_similar_movies(movie_id, 5)
        
        print("\nSimilar movies based on BERT analysis:")
        for similar_id, score in similar_movies:
            similar_title = self.movies_df[self.movies_df["movie_id"] == similar_id].iloc[0]["title"]
            print(f"  - {similar_title} (similarity: {score:.2f})")
        
        print("=" * 60 + "\n")
    
    def get_recommendations(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        compare_with_hybrid: bool = True
    ):
        """
        Get BERT-based recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            n_recommendations: Number of recommendations to generate
            compare_with_hybrid: Whether to compare with hybrid recommendations
        """
        if not self.loaded:
            print("Loading data and building recommenders...")
            self.load_data()
            self.build_recommenders()
            
        # Check if user exists
        if user_id not in self.user_to_idx:
            print(f"User {user_id} not found in the dataset")
            return
        
        # Get user's rated movies
        user_ratings = self.get_user_ratings(user_id)
        user_rated_tuples = [(movie_id, rating) for movie_id, _, rating in user_ratings]
        
        print(f"\nGenerating recommendations for user {user_id}...")
        
        # Get BERT recommendations
        start_time = time.time()
        bert_recs = self.bert_recommender.recommend_for_user(
            user_rated_tuples,
            n_recommendations=n_recommendations
        )
        bert_time = time.time() - start_time
        
        # Print BERT recommendations
        print("\n" + "=" * 80)
        print(f"BERT Recommendations for User {user_id} ({bert_time:.2f} seconds)")
        print("-" * 80)
        
        for i, (movie_id, score) in enumerate(bert_recs, 1):
            movie_info = self.movies_df[self.movies_df["movie_id"] == movie_id]
            movie_title = movie_info.iloc[0]["title"] if len(movie_info) > 0 else f"Movie {movie_id}"
            
            print(f"{i}. {movie_title} (score: {score:.2f})")
            
            # Show explanation
            explanation = self.bert_recommender.explain_recommendation(movie_id, user_rated_tuples)
            
            # Show similar movies from user's ratings
            if explanation["similar_movies"]:
                print("   Similar to movies you've rated:")
                for movie in explanation["similar_movies"][:2]:
                    print(f"   - {movie['title']} (you rated: {movie['rating']:.1f})")
            
            # Show themes
            if explanation["themes"]:
                print("   Themes:")
                for theme, score in explanation["themes"][:2]:
                    theme_name = theme.split(":")[0].strip() if ":" in theme else theme
                    print(f"   - {theme_name}")
            
            print()
        
        # Compare with hybrid recommender if requested
        if compare_with_hybrid and self.hybrid_recommender:
            # Get hybrid recommendations
            start_time = time.time()
            hybrid_recs = self.hybrid_recommender.recommend_for_user(
                user_id,
                n_recommendations=n_recommendations
            )
            hybrid_time = time.time() - start_time
            
            # Print hybrid recommendations
            print("\n" + "=" * 80)
            print(f"Hybrid Recommendations for User {user_id} ({hybrid_time:.2f} seconds)")
            print("-" * 80)
            
            for i, (movie_id, score) in enumerate(hybrid_recs, 1):
                movie_info = self.movies_df[self.movies_df["movie_id"] == movie_id]
                movie_title = movie_info.iloc[0]["title"] if len(movie_info) > 0 else f"Movie {movie_id}"
                
                print(f"{i}. {movie_title} (score: {score:.2f})")
            
            print("\nNote: The BERT recommendations focus on movie content understanding,")
            print("while hybrid recommendations also incorporate collaborative filtering patterns.")
    
    def run_interactive_demo(self):
        """Run an interactive demo session."""
        if not self.loaded:
            print("Loading data and building recommenders...")
            self.load_data()
            self.build_recommenders()
        
        print("\nWelcome to the BERT-based Movie Recommender Demo!")
        print("=" * 60)
        
        while True:
            print("\nOptions:")
            print("1. Show user ratings")
            print("2. Get BERT recommendations")
            print("3. Analyze a movie with BERT")
            print("4. Compare BERT with hybrid recommendations")
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
                n_recommendations = int(input("Number of recommendations (default 10): ") or 10)
                
                self.get_recommendations(
                    user_id, 
                    n_recommendations=n_recommendations,
                    compare_with_hybrid=False
                )
                
            elif choice == "3":
                movie_id = int(input("Enter movie ID: "))
                self.analyze_movie(movie_id)
                
            elif choice == "4":
                user_id = int(input("Enter user ID: "))
                n_recommendations = int(input("Number of recommendations (default 5): ") or 5)
                
                self.get_recommendations(
                    user_id, 
                    n_recommendations=n_recommendations,
                    compare_with_hybrid=True
                )
                
            elif choice == "5":
                print("\nThank you for using the BERT-based Movie Recommender Demo!")
                break
                
            else:
                print("Invalid choice, please try again")

def main():
    """Run the BERT-based Movie Recommendation Demo."""
    parser = argparse.ArgumentParser(description='BERT-based Movie Recommendation Demo')
    
    parser.add_argument('--user', type=int, help='User ID to generate recommendations for')
    parser.add_argument('--movie', type=int, help='Movie ID to analyze with BERT')
    parser.add_argument('--recommendations', type=int, default=10,
                      help='Number of recommendations to generate')
    parser.add_argument('--compare', action='store_true',
                      help='Compare BERT with hybrid recommendations')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                      help='BERT model name to use')
    parser.add_argument('--interactive', action='store_true',
                      help='Run in interactive demo mode')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = BERTMovieLensDemo(bert_model_name=args.model)
    
    if args.interactive:
        # Run interactive demo
        demo.run_interactive_demo()
    elif args.movie:
        # Analyze a specific movie
        demo.analyze_movie(args.movie)
    elif args.user:
        # Generate recommendations for a specific user
        demo.get_recommendations(
            args.user,
            n_recommendations=args.recommendations,
            compare_with_hybrid=args.compare
        )
    else:
        # Default to interactive mode
        demo.run_interactive_demo()

if __name__ == "__main__":
    main() 