"""
Text analysis module using transformer models for movie descriptions.

This module provides functions to analyze movie descriptions using BERT and other
transformer models from the Hugging Face library. It includes sentiment analysis,
theme extraction, and embedding generation for movies.
"""

import numpy as np
import pandas as pd
import torch
import logging
import os
import pickle
from typing import List, Dict, Tuple, Union, Optional, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModel,
    pipeline
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieTextAnalyzer:
    """Class for analyzing movie descriptions using transformer models."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = None,
        cache_dir: str = "models/transformers"
    ):
        """
        Initialize the movie text analyzer.
        
        Args:
            model_name: Name of the transformer model for embeddings
            sentiment_model_name: Name of the sentiment analysis model
            device: Device to run models on ('cpu', 'cuda', or None for auto-detection)
            cache_dir: Directory to cache downloaded models
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_name = model_name
        self.sentiment_model_name = sentiment_model_name
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize models to None (lazy loading)
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        
        logger.info(f"Initialized MovieTextAnalyzer with {model_name} on {self.device}")
    
    def _load_embedding_model(self):
        """Load the transformer model and tokenizer for embeddings."""
        if self.tokenizer is None or self.model is None:
            logger.info(f"Loading transformer model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            ).to(self.device)
        return self.tokenizer, self.model
    
    def _load_sentiment_model(self):
        """Load the sentiment analysis pipeline."""
        if self.sentiment_pipeline is None:
            logger.info(f"Loading sentiment model: {self.sentiment_model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                tokenizer=self.sentiment_model_name,
                device=0 if self.device == "cuda" else -1,
                cache_dir=self.cache_dir
            )
        return self.sentiment_pipeline
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 8,
        max_length: int = 512,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts using the transformer model.
        
        Args:
            texts: List of text strings to generate embeddings for
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            normalize: Whether to L2-normalize the embeddings
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        tokenizer, model = self._load_embedding_model()
        
        # Process in batches to avoid OOM errors
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Skip empty texts
            batch_texts = [text if text and isinstance(text, str) else "" for text in batch_texts]
            
            # Tokenize
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
                
            # Use CLS token embedding as the sentence embedding
            embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        # Normalize embeddings if requested
        if normalize:
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            all_embeddings = all_embeddings / norms
        
        return all_embeddings
    
    def analyze_sentiment(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Analyze sentiment of movie descriptions.
        
        Args:
            texts: List of text strings to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with sentiment scores
        """
        sentiment_pipeline = self._load_sentiment_model()
        
        # Process in batches
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Skip empty texts
            batch_texts = [text if text and isinstance(text, str) else "" for text in batch_texts]
            
            # Make sure all texts are short enough
            batch_texts = [text[:1000] for text in batch_texts]  # Truncate long texts
            
            batch_results = sentiment_pipeline(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def extract_themes(
        self, 
        texts: List[str], 
        n_themes: int = 10
    ) -> Tuple[List[str], np.ndarray]:
        """
        Extract common themes from movie descriptions using embeddings and clustering.
        
        Args:
            texts: List of movie descriptions
            n_themes: Number of themes to extract
            
        Returns:
            Tuple of (list of theme descriptions, array of theme scores per movie)
        """
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.generate_embeddings(texts)
        
        # Apply dimensionality reduction
        logger.info("Applying PCA for dimensionality reduction")
        pca = PCA(n_components=min(100, embeddings.shape[0], embeddings.shape[1]))
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Cluster to find themes
        logger.info(f"Clustering to find {n_themes} themes")
        kmeans = KMeans(n_clusters=n_themes, random_state=42)
        clusters = kmeans.fit_predict(reduced_embeddings)
        
        # Calculate distance to each cluster center
        distances = kmeans.transform(reduced_embeddings)
        
        # Convert distances to similarity scores (1 - normalized distance)
        max_distances = np.max(distances, axis=0)
        similarities = 1 - (distances / max_distances)
        
        # Create theme descriptions
        theme_descriptions = []
        for i in range(n_themes):
            # Get texts closest to this cluster
            cluster_texts = [text for j, text in enumerate(texts) if clusters[j] == i]
            
            # Use most representative texts (up to 3)
            if cluster_texts:
                theme_desc = f"Theme {i+1}: " + " | ".join(cluster_texts[:3])
                theme_descriptions.append(theme_desc)
            else:
                theme_descriptions.append(f"Theme {i+1}: (empty cluster)")
        
        return theme_descriptions, similarities
    
    def process_movie_dataframe(
        self, 
        movies_df: pd.DataFrame,
        description_col: str = 'description',
        title_col: str = 'title',
        save_to_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process movie dataframe to extract embeddings, sentiments, and themes.
        
        Args:
            movies_df: DataFrame with movie data
            description_col: Column name for movie descriptions
            title_col: Column name for movie titles
            save_to_file: Optional path to save results to
            
        Returns:
            Dictionary with text analysis results
        """
        # Check if description column exists
        if description_col not in movies_df.columns:
            # Try to create a simple description from title and genre if available
            if 'genres' in movies_df.columns:
                logger.info(f"No '{description_col}' column found, creating from title and genres")
                movies_df[description_col] = movies_df.apply(
                    lambda row: f"{row[title_col]} is a {' '.join(row['genres'])} movie." 
                    if isinstance(row.get('genres'), list) else row[title_col],
                    axis=1
                )
            else:
                logger.info(f"No '{description_col}' column found, using titles as descriptions")
                movies_df[description_col] = movies_df[title_col]
        
        # Get list of descriptions
        descriptions = movies_df[description_col].fillna("").tolist()
        
        # Process descriptions
        logger.info(f"Processing {len(descriptions)} movie descriptions")
        
        # Generate embeddings
        logger.info("Generating BERT embeddings")
        embeddings = self.generate_embeddings(descriptions)
        
        # Analyze sentiment
        logger.info("Analyzing sentiment")
        sentiment_results = self.analyze_sentiment(descriptions)
        
        # Extract positive sentiment scores
        positive_scores = [
            result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]
            for result in sentiment_results
        ]
        
        # Extract themes (if enough descriptions)
        if len(descriptions) >= 10:
            logger.info("Extracting themes")
            n_themes = min(10, len(descriptions) // 5)  # Adjust number of themes based on data size
            theme_descriptions, theme_scores = self.extract_themes(descriptions, n_themes)
        else:
            theme_descriptions = []
            theme_scores = np.zeros((len(descriptions), 0))
        
        # Compile results
        results = {
            "embeddings": embeddings,
            "positive_sentiment": positive_scores,
            "theme_descriptions": theme_descriptions,
            "theme_scores": theme_scores,
        }
        
        # Save results if requested
        if save_to_file:
            os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
            with open(save_to_file, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Saved text analysis results to {save_to_file}")
        
        return results

def embed_movie_descriptions(
    movies_df: pd.DataFrame,
    description_col: str = 'description',
    model_name: str = "distilbert-base-uncased",
    save_to_file: Optional[str] = None
) -> np.ndarray:
    """
    Generate BERT embeddings for movie descriptions.
    
    Args:
        movies_df: DataFrame with movie data
        description_col: Column name for movie descriptions
        model_name: Name of the BERT model to use
        save_to_file: Optional path to save embeddings to
        
    Returns:
        NumPy array of embeddings with shape (n_movies, embedding_dim)
    """
    analyzer = MovieTextAnalyzer(model_name=model_name)
    
    # Check if description column exists
    if description_col not in movies_df.columns:
        # Use titles as descriptions
        logger.info(f"No '{description_col}' column found, using titles as descriptions")
        descriptions = movies_df['title'].fillna("").tolist()
    else:
        descriptions = movies_df[description_col].fillna("").tolist()
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(descriptions)} movie descriptions")
    embeddings = analyzer.generate_embeddings(descriptions)
    
    # Save embeddings if requested
    if save_to_file:
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        np.save(save_to_file, embeddings)
        logger.info(f"Saved embeddings to {save_to_file}")
    
    return embeddings

if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import load_movielens_100k
    
    # Load data
    data = load_movielens_100k()
    movies_df = data["movies"]
    
    # Create a text analyzer
    analyzer = MovieTextAnalyzer()
    
    # Process movie descriptions (using titles since MovieLens 100K doesn't have descriptions)
    results = analyzer.process_movie_dataframe(
        movies_df, 
        description_col='title',  # Using titles as descriptions
        save_to_file="data/processed/text_analysis_results.pkl"
    )
    
    print(f"Generated {results['embeddings'].shape[1]}-dimensional embeddings for {len(movies_df)} movies")
    print(f"Extracted {len(results['theme_descriptions'])} themes from movie titles") 