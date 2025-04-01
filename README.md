# MovieGen - Movie Recommendation System

A comprehensive movie recommendation system built using the MovieLens dataset. This project implements and compares multiple recommendation algorithms including collaborative filtering, matrix factorization, content-based filtering, and hybrid approaches.

## Project Overview

This system provides movie recommendations based on user preferences using several approaches:

1. **Collaborative Filtering**
   - Item-based: Recommends movies similar to ones the user has liked
   - User-based: Recommends movies liked by similar users

2. **Matrix Factorization**
   - Alternating Least Squares (ALS): Decomposes user-item interaction matrix into latent factors
   - Stochastic Gradient Descent (SGD): Iteratively optimizes latent factors

3. **Content-Based Filtering**
   - Genre-based: Recommends movies with similar genres to favorites
   - Title-based: Uses text similarity of movie titles

4. **Hybrid Recommendation**
   - Combines collaborative filtering, matrix factorization, and content-based approaches
   - Provides explanations for recommendations

## Project Structure

```
MovieGen/
├── data/                     # Data directory
│   ├── raw/                  # Raw MovieLens dataset
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Fully processed data
├── notebooks/                # Jupyter notebooks
│   └── figures/              # Visualization outputs
├── output/                   # Evaluation results and reports
├── src/                      # Source code
│   ├── data/                 # Data loading and processing
│   │   ├── data_loader.py    # Functions to load MovieLens data
│   │   └── download_data.py  # Script to download the dataset
│   ├── features/             # Feature engineering
│   │   └── feature_engineering.py  # Creates matrices and features
│   ├── models/               # Recommendation algorithms
│   │   ├── collaborative_filtering.py  # User/Item-based CF
│   │   ├── matrix_factorization.py     # ALS and SGD
│   │   ├── content_based.py            # Content-based filtering
│   │   └── hybrid_recommender.py       # Combined approach
│   ├── evaluation/           # Evaluation metrics and tools
│   │   └── evaluator.py      # Precision, recall, diversity metrics
│   ├── demo.py               # Interactive demo application
│   └── run_comparison.py     # Compare recommender performance
└── requirements.txt          # Project dependencies
```

## Setup

1. Clone the repository
   ```
   git clone https://github.com/yourusername/MovieGen.git
   cd MovieGen
   ```

2. Create a virtual environment (optional but recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Download the MovieLens dataset
   ```
   python src/data/download_data.py
   ```

## Usage

### Interactive Demo

Run the interactive command-line demo to get movie recommendations:

```
python src/demo.py --interactive
```

Or get recommendations for a specific user:

```
python src/demo.py --user 42 --algorithm hybrid --recommendations 10 --explain
```

Command-line options:
- `--user`: User ID to generate recommendations for
- `--algorithm`: Recommendation algorithm to use (itemcf, usercf, als, content, hybrid)
- `--recommendations`: Number of recommendations to generate (default: 10)
- `--explain`: Show explanation for recommendations (only for hybrid)
- `--interactive`: Run in interactive mode

### Evaluate and Compare Models

Run a comparison of all recommendation models:

```
python src/run_comparison.py
```

This will:
1. Train all recommendation models on the MovieLens dataset
2. Evaluate their performance using various metrics (precision, recall, diversity)
3. Generate charts comparing the models
4. Save example recommendations for sample users
5. Output results to the `output/` directory

## Recommendation Approaches

### Collaborative Filtering

- **Item-based Collaborative Filtering**: Finds similarities between movies based on user ratings and recommends similar movies to what a user has liked.
- **User-based Collaborative Filtering**: Finds similar users based on rating patterns and recommends movies that similar users have liked.

### Matrix Factorization

- **Alternating Least Squares (ALS)**: Decomposes the user-item interaction matrix into user and item latent factors, alternating between fixing one and optimizing the other.
- **Stochastic Gradient Descent (SGD)**: Iteratively optimizes the user and item latent factors to minimize prediction error.

### Content-Based Filtering

- **Genre-based Recommender**: Recommends movies with similar genres to what a user has liked.
- **Title-based Recommender**: Uses text similarity between movie titles to find similar movies.
- **Hybrid Content Recommender**: Combines genre and title similarity with configurable weights.

### Hybrid Recommender

The hybrid recommender combines:
- Item-based collaborative filtering
- Matrix factorization (ALS)
- Content-based filtering

It provides explanations for why a movie was recommended, such as similar genres to movies you've enjoyed or similarity to movies you've rated highly.

## Evaluation Metrics

The system evaluates recommendations using:

- **Rating Prediction**: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)
- **Ranking Quality**: Precision@k, Recall@k, F1@k
- **Diversity**: Ratio of unique items in recommendations
- **Execution Time**: Time taken to generate recommendations

## Example Output

The evaluation produces:
- CSV files with comparison results
- Bar charts for each metric
- A heatmap comparing all models and metrics
- Example recommendations for sample users in HTML format
- Detailed explanations for hybrid recommendations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT 