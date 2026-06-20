# src/data_preprocessing.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer  # ← Change this back
import os

# Download required NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text: str) -> str:
    """
    Clean raw text by removing noise.
    
    Args:
        text: Raw review text
    
    Returns:
        Cleaned text (lowercase, no HTML/URLs/punctuation/numbers)
    """
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove HTML tags (e.g., <br />)
    text = re.sub(r'<.*?>', ' ', text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    
    # 4. Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 5. Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # 6. Remove extra whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text: str, apply_stemming: bool = True) -> str:
    """
    Full preprocessing pipeline: clean -> tokenize -> remove stopwords -> stem.
    
    Args:
        text: Raw review text
        apply_stemming: If True, apply Porter Stemmer. If False, return tokens as string.
    
    Returns:
        Preprocessed text as a single string (tokens joined by space)
    """
    # Step 1: Clean the text
    cleaned = clean_text(text)
    
    # Step 2: Tokenize (split into words)
    tokens = word_tokenize(cleaned)
    
    # Step 3: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Step 4: Apply Stemming (Porter Stemmer)
    if apply_stemming:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    
    # Step 5: Join tokens back into a single string
    return ' '.join(tokens)

def preprocess_dataset(df: pd.DataFrame, text_column: str = 'review', 
                       apply_stemming: bool = True) -> pd.DataFrame:
    """
    Preprocess the entire dataset.
    
    Args:
        df: DataFrame with raw reviews
        text_column: Name of column containing text
        apply_stemming: Whether to apply lemmatization
    
    Returns:
        DataFrame with additional 'cleaned_review' column
    """
    print("\n=== Starting Text Preprocessing ===")
    print(f"Original dataset shape: {df.shape}")
    
    # Check for duplicates and remove them
    initial_count = len(df)
    df = df.drop_duplicates(subset=[text_column])
    print(f"Removed {initial_count - len(df)} duplicate rows")
    
    # Apply preprocessing to each review
    print("Preprocessing reviews... (this may take 1-2 minutes)")
    df['cleaned_review'] = df[text_column].apply(
        lambda x: preprocess_text(x, apply_stemming=apply_stemming)
    )
    
    # Remove any empty reviews after preprocessing
    df = df[df['cleaned_review'].str.len() > 0]
    print(f"Final dataset shape: {df.shape}")
    
    return df

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the IMDB dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file (e.g., 'data/raw/IMDB Dataset.csv')
    
    Returns:
        DataFrame with columns: review, sentiment
    """
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Display basic info
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return df

def save_processed_data(df: pd.DataFrame, save_path: str = "data/processed/cleaned_data.csv"):
    """
    Save the preprocessed dataset to CSV.
    
    Args:
        df: DataFrame with cleaned reviews
        save_path: Path to save the CSV file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n✅ Cleaned data saved to {save_path}")

if __name__ == "__main__":
    # Quick test when run directly
    file_path = "data/raw/IMDB Dataset.csv"
    if os.path.exists(file_path):
        df = load_data(file_path)
        
        # Show a before/after example
        sample = df['review'].iloc[0]
        print("\n=== Before Preprocessing ===")
        print(sample[:200] + "...")
        
        cleaned = preprocess_text(sample)
        print("\n=== After Preprocessing (Lemmatization) ===")
        print(cleaned[:200] + "...")
        
        # Preprocess full dataset
        df_processed = preprocess_dataset(df)
        save_processed_data(df_processed)
    else:
        print(f"Error: File not found at {file_path}")