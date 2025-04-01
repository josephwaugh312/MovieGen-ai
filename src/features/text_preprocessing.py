"""
Text preprocessing utilities for natural language processing tasks.
"""
import re
import string
import logging
from typing import List, Set, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
def download_nltk_resources():
    """Download necessary NLTK resources if they're not already downloaded."""
    nltk_resources = {'tokenizers/punkt': 'punkt',
                    'corpora/stopwords': 'stopwords',
                    'corpora/wordnet': 'wordnet'}
    
    for resource_path, resource_name in nltk_resources.items():
        try:
            nltk.data.find(resource_path)
            logger.info(f"NLTK resource already downloaded: {resource_name}")
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource_name}")
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {resource_name}: {e}")

# Download resources on module import
download_nltk_resources()

def get_stopwords(extra_stopwords: Optional[List[str]] = None) -> Set[str]:
    """
    Get a set of stopwords.
    
    Args:
        extra_stopwords (List[str], optional): Additional stopwords to include
        
    Returns:
        Set[str]: Set of stopwords
    """
    # Get English stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        logger.warning("Stopwords not available. Using a basic set instead.")
        # Basic set of English stopwords if NLTK's stopwords aren't available
        stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 
                        'as', 'what', 'which', 'this', 'that', 'these', 'those', 
                        'then', 'just', 'so', 'than', 'such', 'both', 'through', 
                        'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from'])
    
    # Add extra stopwords if provided
    if extra_stopwords:
        stop_words.update(extra_stopwords)
        
    return stop_words

def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase and removing special characters.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and replace with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenization by splitting on whitespace.
    Used as a fallback when NLTK tokenization fails.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of tokens
    """
    return text.split()

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of tokens
    """
    try:
        return word_tokenize(text)
    except LookupError as e:
        logger.warning(f"NLTK tokenization failed: {e}. Using simple tokenization instead.")
        return simple_tokenize(text)

def remove_stopwords(tokens: List[str], stop_words: Optional[Set[str]] = None) -> List[str]:
    """
    Remove stopwords from a list of tokens.
    
    Args:
        tokens (List[str]): List of tokens
        stop_words (Set[str], optional): Set of stopwords to remove
        
    Returns:
        List[str]: Filtered tokens
    """
    if stop_words is None:
        stop_words = get_stopwords()
        
    return [token for token in tokens if token not in stop_words]

def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Apply stemming to tokens.
    
    Args:
        tokens (List[str]): List of tokens
        
    Returns:
        List[str]: Stemmed tokens
    """
    try:
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]
    except Exception as e:
        logger.warning(f"Stemming failed: {e}. Returning original tokens.")
        return tokens

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Apply lemmatization to tokens.
    
    Args:
        tokens (List[str]): List of tokens
        
    Returns:
        List[str]: Lemmatized tokens
    """
    try:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    except Exception as e:
        logger.warning(f"Lemmatization failed: {e}. Returning original tokens.")
        return tokens

def preprocess_text(
    text: str,
    remove_stopwords_flag: bool = True,
    stem: bool = False,
    lemmatize: bool = True,
    extra_stopwords: Optional[List[str]] = None
) -> List[str]:
    """
    Preprocess text by applying multiple preprocessing steps.
    
    Args:
        text (str): Input text
        remove_stopwords_flag (bool): Whether to remove stopwords
        stem (bool): Whether to apply stemming
        lemmatize (bool): Whether to apply lemmatization
        extra_stopwords (List[str], optional): Additional stopwords to remove
        
    Returns:
        List[str]: Preprocessed tokens
    """
    # Normalize text
    normalized_text = normalize_text(text)
    
    # Tokenize text
    tokens = tokenize_text(normalized_text)
    
    # Remove stopwords if flag is set
    if remove_stopwords_flag:
        stop_words = get_stopwords(extra_stopwords)
        tokens = remove_stopwords(tokens, stop_words)
    
    # Apply stemming if flag is set
    if stem:
        tokens = stem_tokens(tokens)
    
    # Apply lemmatization if flag is set
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    
    return tokens

def create_bag_of_words(tokens: List[str]) -> dict:
    """
    Create a bag of words representation (word frequency dictionary).
    
    Args:
        tokens (List[str]): List of tokens
        
    Returns:
        dict: Dictionary with word frequencies
    """
    bag_of_words = {}
    for token in tokens:
        bag_of_words[token] = bag_of_words.get(token, 0) + 1
    
    return bag_of_words

def extract_movie_title_features(title: str) -> dict:
    """
    Extract features from a movie title.
    
    Args:
        title (str): Movie title
        
    Returns:
        dict: Dictionary with extracted features
    """
    features = {}
    
    # Extract year from title if present (in format "Movie Title (YYYY)")
    year_match = re.search(r'\((\d{4})\)', title)
    if year_match:
        features['year'] = int(year_match.group(1))
        # Remove year from title for further processing
        clean_title = re.sub(r'\s*\(\d{4}\)', '', title)
    else:
        features['year'] = None
        clean_title = title
    
    # Preprocess title
    tokens = preprocess_text(clean_title)
    
    # Add tokens and token count as features
    features['tokens'] = tokens
    features['token_count'] = len(tokens)
    
    # Create bag of words
    features['bag_of_words'] = create_bag_of_words(tokens)
    
    return features

if __name__ == "__main__":
    # Example usage
    sample_text = "The Shawshank Redemption (1994) is one of the greatest movies of all time!"
    
    # Apply full preprocessing pipeline
    tokens = preprocess_text(sample_text)
    print(f"Preprocessed tokens: {tokens}")
    
    # Extract features from a movie title
    features = extract_movie_title_features(sample_text)
    print(f"Extracted features: {features}")
    
    # Test different preprocessing options
    print("\nDifferent preprocessing options:")
    print(f"Original text: {sample_text}")
    print(f"Normalized: {normalize_text(sample_text)}")
    print(f"Tokenized: {tokenize_text(normalize_text(sample_text))}")
    print(f"Without stopwords: {remove_stopwords(tokenize_text(normalize_text(sample_text)))}")
    print(f"Stemmed: {stem_tokens(remove_stopwords(tokenize_text(normalize_text(sample_text))))}")
    print(f"Lemmatized: {lemmatize_tokens(remove_stopwords(tokenize_text(normalize_text(sample_text))))}") 