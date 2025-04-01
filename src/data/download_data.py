"""
Script to download MovieLens dataset and save it to the data directory.
"""
import os
import zipfile
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_DIR / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs
# MovieLens 100K dataset (smaller, good for development)
ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
# MovieLens 25M dataset (larger, for production)
ML_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

def download_file(url, local_path):
    """
    Download a file from a URL to a local path.
    
    Args:
        url (str): URL of the file to download
        local_path (Path): Local path to save the file
    
    Returns:
        Path: Path to the downloaded file
    """
    if local_path.exists():
        logger.info(f"File already exists at {local_path}. Skipping download.")
        return local_path
    
    logger.info(f"Downloading {url} to {local_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    logger.info(f"Download complete: {local_path}")
    return local_path

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a directory.
    
    Args:
        zip_path (Path): Path to the zip file
        extract_to (Path): Directory to extract to
    """
    logger.info(f"Extracting {zip_path} to {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Extraction complete: {extract_to}")

def download_dataset(dataset_size="small"):
    """
    Download and extract the MovieLens dataset.
    
    Args:
        dataset_size (str): Size of the dataset to download ('small' or 'large')
    
    Returns:
        Path: Path to the extracted dataset directory
    """
    if dataset_size.lower() == "small":
        url = ML_100K_URL
        zip_filename = "ml-100k.zip"
        dataset_dirname = "ml-100k"
    else:
        url = ML_25M_URL
        zip_filename = "ml-25m.zip"
        dataset_dirname = "ml-25m"
    
    zip_path = RAW_DATA_DIR / zip_filename
    
    # Download dataset
    download_file(url, zip_path)
    
    # Extract dataset
    extract_zip(zip_path, RAW_DATA_DIR)
    
    return RAW_DATA_DIR / dataset_dirname

if __name__ == "__main__":
    # Download and extract smaller dataset by default
    dataset_path = download_dataset("small")
    logger.info(f"MovieLens dataset downloaded and extracted to {dataset_path}")
    
    # Print available files
    logger.info("Available files:")
    for file_path in dataset_path.iterdir():
        if file_path.is_file():
            logger.info(f"  - {file_path.name}") 