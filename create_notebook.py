"""
Script to create a valid Jupyter notebook for data exploration.
"""
import json
import os

# Ensure the notebooks directory exists
os.makedirs('notebooks', exist_ok=True)

# Define the notebook content
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MovieLens Dataset Exploration\n",
                "\n",
                "This notebook explores the MovieLens dataset to understand its structure and content."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import necessary libraries\n",
                "import sys\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "\n",
                "# Configure matplotlib and seaborn\n",
                "%matplotlib inline\n",
                "sns.set_style(\"whitegrid\")\n",
                "plt.rcParams['figure.figsize'] = (12, 8)\n",
                "\n",
                "# Add the project directory to the Python path to import custom modules\n",
                "project_dir = Path.cwd().parent\n",
                "sys.path.append(str(project_dir))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import custom modules from the project\n",
                "from src.data.data_loader import load_movielens_100k, get_movie_genres\n",
                "from src.features.text_preprocessing import preprocess_text, extract_movie_title_features"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load the Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the MovieLens 100K dataset\n",
                "try:\n",
                "    data = load_movielens_100k()\n",
                "    # Display the keys in the data dictionary\n",
                "    print(\"Data components:\", list(data.keys()))\n",
                "except FileNotFoundError:\n",
                "    print(\"Dataset not found. Please run 'python src/data/download_data.py' first.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Explore the Users Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display the first few rows of the users dataframe\n",
                "try:\n",
                "    print(\"Users dataset shape:\", data[\"users\"].shape)\n",
                "    data[\"users\"].head()\n",
                "except NameError:\n",
                "    print(\"Please load the dataset first.\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebook to file
with open('notebooks/data_exploration.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Created notebooks/data_exploration.ipynb")
print("You can now run: jupyter lab notebooks/data_exploration.ipynb") 