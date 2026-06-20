# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import joblib

def create_tfidf_features(df: pd.DataFrame,
                          text_column: str = 'cleaned_review',
                          max_features: int = 10000) -> tuple:
    """
    Create TF-IDF features with unigrams and bigrams.
    """
    print("\n=== Creating TF-IDF Features ===")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),  # Unigrams + Bigrams
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    X_tfidf = vectorizer.fit_transform(df[text_column])
    
    print(f"✅ TF-IDF shape: {X_tfidf.shape}")
    print(f"✅ Number of features: {len(vectorizer.get_feature_names_out())}")
    print(f"✅ Sample features (first 10): {vectorizer.get_feature_names_out()[:10].tolist()}")
    
    return X_tfidf, vectorizer

def save_vectorizer(vectorizer, name: str, save_dir: str = "models/"):
    """Save the fitted vectorizer to disk."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{name}.pkl")
    joblib.dump(vectorizer, filepath)
    print(f"✅ Vectorizer saved to {filepath}")

def prepare_features(df: pd.DataFrame, use_tfidf: bool = True, 
                     max_features: int = 10000) -> tuple:
    """
    Complete feature engineering pipeline.
    """
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)
    
    # Encode labels: positive -> 1, negative -> 0
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Create TF-IDF features
    if use_tfidf:
        print("\n🔹 Using TF-IDF with bigrams")
        X, vectorizer = create_tfidf_features(df, text_column='cleaned_review', 
                                               max_features=max_features)
    else:
        print("\n🔹 Using Bag of Words")
        X, vectorizer = create_bow_features(df, text_column='cleaned_review',
                                             max_features=max_features)
    
    # Save the vectorizer
    vectorizer_name = 'tfidf_vectorizer' if use_tfidf else 'bow_vectorizer'
    save_vectorizer(vectorizer, vectorizer_name)
    
    print(f"\n✅ Final feature matrix shape: {X.shape}")
    print(f"✅ Target labels shape: {y.shape}")
    
    return X, y, vectorizer

def create_bow_features(df: pd.DataFrame, 
                        text_column: str = 'cleaned_review',
                        max_features: int = 5000) -> tuple:
    """Create Bag of Words features."""
    print("\n=== Creating Bag of Words Features ===")
    
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 1)
    )
    
    X_bow = vectorizer.fit_transform(df[text_column])
    
    print(f"✅ BoW shape: {X_bow.shape}")
    print(f"✅ Number of features: {len(vectorizer.get_feature_names_out())}")
    
    return X_bow, vectorizer

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_data.csv")
    X, y, vectorizer = prepare_features(df, use_tfidf=True, max_features=10000)