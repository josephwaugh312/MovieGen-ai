"""
Test script to verify the environment is set up correctly.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn import datasets
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    print(f"Seaborn version: {sns.__version__}")
    print(f"NLTK version: {nltk.__version__}")
    
    # Test project imports
    try:
        from src.data import download_data
        from src.features import text_preprocessing
        print("Project imports successful!")
    except ImportError as e:
        print(f"Project import error: {e}")

def test_nltk():
    """Test NLTK functionality."""
    from src.features.text_preprocessing import preprocess_text
    
    sample_text = "Testing the movie recommendation system!"
    tokens = preprocess_text(sample_text)
    print(f"NLTK test - processed '{sample_text}' into tokens: {tokens}")

def test_project_structure():
    """Test that the project structure is set up correctly."""
    project_dir = Path.cwd()
    
    expected_dirs = [
        project_dir / "data" / "raw",
        project_dir / "data" / "processed",
        project_dir / "data" / "interim",
        project_dir / "src" / "data",
        project_dir / "src" / "features",
        project_dir / "src" / "models",
        project_dir / "src" / "visualization",
        project_dir / "notebooks",
    ]
    
    for directory in expected_dirs:
        exists = directory.exists()
        print(f"{directory}: {'✓' if exists else '✗'}")

if __name__ == "__main__":
    print("=== Testing imports ===")
    test_imports()
    
    print("\n=== Testing project structure ===")
    test_project_structure()
    
    print("\n=== Testing NLTK functionality ===")
    test_nltk()
    
    print("\nIf all tests passed, your environment is set up correctly!") 