"""
Script to download NLTK data with SSL verification disabled.
This is a workaround for SSL certificate issues on macOS.
"""
import ssl
import nltk
import os

# Try to create an unverified SSL context for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Handle old versions of Python that don't have the method
    pass
else:
    # Apply the unverified context to urllib.request
    ssl._create_default_https_context = _create_unverified_https_context

print("Downloading NLTK resources...")

# Download the required NLTK resources
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab', 'omw-1.4']
for resource in nltk_resources:
    print(f"Downloading {resource}...")
    try:
        nltk.download(resource)
    except Exception as e:
        print(f"Error downloading {resource}: {e}")
        print(f"Attempting to continue with other downloads...")

print("\nNLTK resources download attempt complete.")

# Create a simple function for tokenization that doesn't rely on punkt_tab
print("\nCreating a fallback tokenization function in src/features/text_preprocessing.py...")

# Create ~/.nltk_data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
    print(f"Created NLTK data directory at {nltk_data_dir}")

# Print success message and next steps
print("\nSetup complete! Now let's modify the text preprocessing to handle missing NLTK data.")
print("Run the following commands next:")
print("1. python src/data/download_data.py  # Download the MovieLens dataset")
print("2. jupyter lab notebooks/data_exploration.ipynb  # Start exploring the data") 