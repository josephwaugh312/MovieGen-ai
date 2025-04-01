"""
Run a comparison of all recommender models and generate a report.

This script loads the MovieLens dataset, trains multiple recommender models,
compares their performance, and generates a report with recommendation examples.
"""

import sys
import os
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from typing import Dict, List, Tuple

# Import project modules
from src.data.data_loader import load_movielens_100k, get_movie_genres
from src.features.feature_engineering import (
    create_user_item_matrix, 
    split_train_test_by_time
)
from src.models.collaborative_filtering import ItemBasedCF, UserBasedCF
from src.models.matrix_factorization import ALS, SGD
from src.models.content_based import (
    GenreBasedRecommender, 
    TitleBasedRecommender,
    HybridContentRecommender
)
from src.models.hybrid_recommender import HybridRecommender, get_recommendation_explanation
from src.evaluation.evaluator import (
    RecommenderEvaluator,
    create_test_items_for_users,
    compare_recommenders
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_dir(dir_name: str = "output") -> str:
    """Create output directory if it doesn't exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def save_comparison_chart(results_df: pd.DataFrame, metrics: List[str], output_dir: str):
    """
    Create and save bar charts comparing different recommenders on various metrics.
    
    Args:
        results_df: DataFrame with comparison results
        metrics: List of metrics to include in chart
        output_dir: Directory to save chart
    """
    for metric in metrics:
        if metric in results_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Sort by metric value
            sorted_df = results_df.sort_values(by=metric)
            
            # Create bar chart
            ax = sorted_df[metric].plot(kind='barh')
            
            # Add values to end of bars
            for i, v in enumerate(sorted_df[metric]):
                if not np.isnan(v):
                    ax.text(v + 0.01, i, f"{v:.4f}", va='center')
            
            # Set labels and title
            plt.xlabel(f"{metric} value")
            plt.ylabel("Recommender")
            plt.title(f"Comparison of Recommenders - {metric}")
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"comparison_{metric}.png"))
            plt.close()
            
    # Save overall metrics as heatmap if there are multiple metrics
    if len(metrics) > 1:
        plt.figure(figsize=(12, 8))
        
        # Normalize each metric column to 0-1 range
        norm_df = results_df.copy()
        for col in metrics:
            if col in norm_df.columns:
                col_min = norm_df[col].min()
                col_max = norm_df[col].max()
                if col_max > col_min:
                    norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        
        # Create heatmap
        plt.imshow(norm_df[metrics].T, cmap='viridis')
        
        # Add labels
        plt.yticks(range(len(metrics)), metrics)
        plt.xticks(range(len(norm_df)), norm_df.index, rotation=45, ha='right')
        plt.colorbar(label='Normalized Score')
        
        # Add values in cells
        for i in range(len(norm_df)):
            for j, metric in enumerate(metrics):
                if metric in results_df.columns:
                    plt.text(i, j, f"{results_df.iloc[i][metric]:.3f}", 
                            ha="center", va="center", color="white")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_heatmap.png"))
        plt.close()

def generate_example_recommendations(
    recommenders: Dict[str, any],
    sample_users: List[int],
    movies_df: pd.DataFrame,
    movie_genres_df: pd.DataFrame,
    n_recommendations: int = 5,
    output_dir: str = "output"
):
    """
    Generate and save example recommendations for sample users.
    
    Args:
        recommenders: Dictionary mapping recommender name to recommender object
        sample_users: List of user IDs to generate recommendations for
        movies_df: DataFrame with movie metadata
        movie_genres_df: DataFrame with movie genres
        n_recommendations: Number of recommendations to generate
        output_dir: Directory to save example recommendations
    """
    # Create DataFrame to store all recommendations
    all_recs = []
    
    for user_id in sample_users:
        logger.info(f"Generating recommendations for user {user_id}")
        
        # Get user's age and occupation
        for recommender_name, recommender in recommenders.items():
            try:
                # Get recommendations for this user
                user_recs = recommender.recommend_for_user(
                    user_id, 
                    n_recommendations=n_recommendations,
                    exclude_rated=True
                )
                
                for rank, (movie_id, score) in enumerate(user_recs, 1):
                    # Get movie title and other info
                    movie_info = movies_df[movies_df["movie_id"] == movie_id]
                    if len(movie_info) > 0:
                        movie_title = movie_info.iloc[0]["title"]
                        
                        # Get movie genres
                        movie_genres = []
                        for _, row in movie_genres_df.iterrows():
                            if row["movie_id"] == movie_id:
                                movie_genres = row["genres"]
                                break
                        
                        # Get explanation if available
                        explanation = ""
                        if recommender_name == "Hybrid" and hasattr(recommender, "explain_recommendation"):
                            try:
                                explanation = get_recommendation_explanation(
                                    user_id, 
                                    movie_id, 
                                    {"movies": movies_df}, 
                                    recommender,
                                    movie_genres_df
                                )
                            except Exception as e:
                                logger.warning(f"Error generating explanation: {e}")
                                explanation = ""
                        
                        all_recs.append({
                            "user_id": user_id,
                            "recommender": recommender_name,
                            "rank": rank,
                            "movie_id": movie_id,
                            "movie_title": movie_title,
                            "genres": ", ".join(movie_genres) if movie_genres else "",
                            "score": score,
                            "explanation": explanation
                        })
            
            except Exception as e:
                logger.warning(f"Error generating recommendations with {recommender_name} for user {user_id}: {e}")
    
    # Create DataFrame and save to CSV
    recs_df = pd.DataFrame(all_recs)
    csv_path = os.path.join(output_dir, "example_recommendations.csv")
    recs_df.to_csv(csv_path, index=False)
    logger.info(f"Saved example recommendations to {csv_path}")
    
    # Create a formatted HTML report
    html_path = os.path.join(output_dir, "recommendation_examples.html")
    
    with open(html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MovieLens Recommendation Examples</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .user-section { margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }
                .recommender-section { margin-bottom: 20px; }
                .recommendation { margin-bottom: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }
                .movie-title { font-weight: bold; }
                .genres { font-style: italic; color: #666; }
                .score { color: #0066cc; }
                .explanation { margin-top: 5px; padding: 5px; background-color: #efefef; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <h1>MovieLens Recommendation Examples</h1>
        """)
        
        # Group by user
        for user_id in sample_users:
            user_recs = recs_df[recs_df["user_id"] == user_id]
            
            f.write(f"<div class='user-section'>\n")
            f.write(f"<h2>User {user_id}</h2>\n")
            
            # Group by recommender
            for recommender_name, group in user_recs.groupby("recommender"):
                f.write(f"<div class='recommender-section'>\n")
                f.write(f"<h3>{recommender_name} Recommendations</h3>\n")
                
                # Sort by rank
                for _, row in group.sort_values("rank").iterrows():
                    f.write(f"<div class='recommendation'>\n")
                    f.write(f"<div class='movie-title'>{row['rank']}. {row['movie_title']}</div>\n")
                    f.write(f"<div class='genres'>Genres: {row['genres']}</div>\n")
                    f.write(f"<div class='score'>Score: {row['score']:.4f}</div>\n")
                    
                    if row['explanation']:
                        f.write(f"<div class='explanation'>{row['explanation'].replace('•', '&bull;').replace('\n', '<br>')}</div>\n")
                    
                    f.write("</div>\n")
                
                f.write("</div>\n")
            
            f.write("</div>\n")
        
        f.write("""
        </body>
        </html>
        """)
    
    logger.info(f"Saved HTML recommendation examples to {html_path}")

def main():
    """Run the main comparison of recommender systems."""
    start_time = time.time()
    logger.info("Starting recommender system comparison")
    
    # Create output directory
    output_dir = create_output_dir("output")
    
    # Load data
    logger.info("Loading MovieLens dataset")
    data = load_movielens_100k()
    
    # Create movie genres DataFrame
    movie_genres_df = get_movie_genres(data["movies"], data["genres"])
    
    # Split into train and test
    logger.info("Splitting data into train and test sets")
    train_ratings, test_ratings = split_train_test_by_time(data["ratings"])
    
    # Create user-item matrix for training
    logger.info("Creating user-item matrix")
    full_matrix, idx_to_user, idx_to_movie, user_to_idx, movie_to_idx = create_user_item_matrix(
        data["ratings"]
    )
    
    train_matrix, train_idx_to_user, train_idx_to_movie, train_user_to_idx, train_movie_to_idx = create_user_item_matrix(
        train_ratings
    )
    
    # Create test items for evaluation
    logger.info("Creating test data for evaluation")
    test_relevant_items, test_user_ratings = create_test_items_for_users(
        test_ratings, 
        train_ratio=0, 
        rating_threshold=4.0,
        min_ratings=5
    )
    
    # List of all movie IDs
    all_movies = data["movies"]["movie_id"].tolist()
    
    # Build all recommenders
    logger.info("Building and fitting recommender models")
    recommenders = {}
    
    # Add item-based collaborative filtering
    logger.info("Building ItemBasedCF")
    recommenders["ItemBasedCF"] = ItemBasedCF(
        similarity_method='cosine',
        min_support=2,
        k_neighbors=30
    ).fit(
        train_matrix, 
        train_idx_to_user, 
        train_idx_to_movie, 
        train_user_to_idx, 
        train_movie_to_idx
    )
    
    # Add user-based collaborative filtering
    logger.info("Building UserBasedCF")
    recommenders["UserBasedCF"] = UserBasedCF(
        similarity_method='pearson',
        min_support=2,
        k_neighbors=30
    ).fit(
        train_matrix, 
        train_idx_to_user, 
        train_idx_to_movie, 
        train_user_to_idx, 
        train_movie_to_idx
    )
    
    # Add matrix factorization
    logger.info("Building ALS")
    recommenders["ALS"] = ALS(
        n_factors=50,
        regularization=0.1,
        n_iterations=15
    ).fit(
        train_matrix, 
        train_idx_to_user, 
        train_idx_to_movie, 
        train_user_to_idx, 
        train_movie_to_idx
    )
    
    # Add content-based recommenders
    logger.info("Building GenreBased")
    genre_recommender = GenreBasedRecommender()
    genre_recommender.fit(data["movies"], movie_genres_df)
    recommenders["GenreBased"] = genre_recommender
    
    logger.info("Building HybridContent")
    content_recommender = HybridContentRecommender(
        genre_weight=0.7,
        title_weight=0.3
    )
    content_recommender.fit(data["movies"], movie_genres_df)
    recommenders["HybridContent"] = content_recommender
    
    # Add hybrid recommender
    logger.info("Building Hybrid")
    hybrid_recommender = HybridRecommender(
        item_cf_weight=0.4,
        mf_weight=0.3,
        content_weight=0.3,
        n_factors=50,
        item_cf_min_support=2,
        item_cf_k_neighbors=30
    )
    
    hybrid_recommender.fit(
        train_matrix,
        train_idx_to_user,
        train_idx_to_movie,
        train_user_to_idx,
        train_movie_to_idx,
        data["movies"],
        movie_genres_df,
        full_matrix
    )
    
    recommenders["Hybrid"] = hybrid_recommender
    
    # Compare recommenders
    logger.info("Comparing recommenders")
    metrics = [
        'precision@5', 
        'precision@10', 
        'recall@10', 
        'diversity',
        'execution_time'
    ]
    
    results = compare_recommenders(
        recommenders,
        test_user_ratings,
        all_movies,
        metrics=metrics,
        n_recommendations=10
    )
    
    # Save results to CSV
    results_csv_path = os.path.join(output_dir, "recommender_comparison.csv")
    results.to_csv(results_csv_path)
    logger.info(f"Saved comparison results to {results_csv_path}")
    
    # Create and save comparison charts
    save_comparison_chart(results, metrics, output_dir)
    
    # Generate example recommendations for a few sample users
    sample_users = [1, 42, 100, 233, 453]  # Adjust with actual user IDs from your dataset
    generate_example_recommendations(
        recommenders,
        sample_users,
        data["movies"],
        movie_genres_df,
        n_recommendations=5,
        output_dir=output_dir
    )
    
    # Print final results
    print("\nRecommender System Comparison Results:")
    print(results.round(4))
    
    # Log execution time
    execution_time = time.time() - start_time
    logger.info(f"Completed comparison in {execution_time:.2f} seconds")
    print(f"\nComparison completed in {execution_time:.2f} seconds")
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main() 