# main.py
import sys
import os
import pandas as pd
import numpy as np
import joblib
from src.data_preprocessing import load_data, preprocess_dataset, save_processed_data, preprocess_text
from src.eda import run_eda
from src.feature_engineering import prepare_features
from src.train_model import train_models, save_models
from src.evaluate_model import evaluate_models
from src.tune_model import tune_logistic_regression, evaluate_tuned_model, save_tuned_model

def interactive_mode():
    """Run interactive sentiment prediction."""
    print("\n" + "="*50)
    print("🔮 SENTIMENT ANALYSIS INTERACTIVE PREDICTOR")
    print("="*50)
    
    model_path = "models/logistic_regression_tuned.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please run full pipeline first.")
        return
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model and vectorizer loaded successfully!")
    
    print("\nType a movie review and get sentiment prediction instantly.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("📝 Enter your review: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            print("⚠️  Please enter a review.\n")
            continue
        
        try:
            cleaned = preprocess_text(user_input, apply_stemming=True)
            text_tfidf = vectorizer.transform([cleaned])
            prediction = model.predict(text_tfidf)[0]
            sentiment = 'Positive' if prediction == 1 else 'Negative'
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(text_tfidf)[0]
                confidence = proba[1] if prediction == 1 else proba[0]
            else:
                decision = model.decision_function(text_tfidf)[0]
                confidence = 1 / (1 + np.exp(-decision))
                if prediction == 0:
                    confidence = 1 - confidence
            
            print("\n" + "-"*40)
            print(f"📝 Review: {user_input[:100]}...")
            print(f"🧹 Cleaned: {cleaned[:100]}...")
            print(f"🎯 Sentiment: {sentiment}")
            print(f"📊 Confidence: {confidence*100:.1f}%")
            print("-"*40 + "\n")
            
        except Exception as e:
            print(f"⚠️  Error: {str(e)}\n")

def quick_test():
    """Run quick test on sample reviews."""
    print("\n=== Testing Sample Reviews ===\n")
    
    model_path = "models/logistic_regression_tuned.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please run full pipeline first.")
        return
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    sample_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Worst film I've ever seen. Complete waste of time.",
        "It was okay, nothing special but not terrible either.",
        "The acting was brilliant and the story kept me hooked.",
        "I fell asleep halfway through. Boring and predictable."
    ]
    
    for review in sample_reviews:
        cleaned = preprocess_text(review, apply_stemming=True)
        text_tfidf = vectorizer.transform([cleaned])
        prediction = model.predict(text_tfidf)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_tfidf)[0]
            confidence = proba[1] if prediction == 1 else proba[0]
        else:
            decision = model.decision_function(text_tfidf)[0]
            confidence = 1 / (1 + np.exp(-decision))
            if prediction == 0:
                confidence = 1 - confidence
        
        print(f"Review: {review[:60]}...")
        print(f"Sentiment: {sentiment} (Confidence: {confidence*100:.1f}%)\n")

def main():
    print("=== Sentiment Analysis Pipeline ===\n")
    print("Choose mode:")
    print("  1. Full pipeline (train models)")
    print("  2. Interactive prediction")
    print("  3. Quick test on sample reviews")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == '2':
        interactive_mode()
        return
    
    if choice == '3':
        quick_test()
        return
    
    if choice != '1':
        print("Invalid choice. Running full pipeline...")
    
    # ===== FULL PIPELINE =====
    
    data_path = "data/raw/IMDB Dataset.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)
    
    df = load_data(data_path)
    
    print("\n=== Basic Data Inspection ===")
    print(f"Total samples: {len(df)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    print("\n=== Class Distribution ===")
    print(df['sentiment'].value_counts())
    print(df['sentiment'].value_counts(normalize=True) * 100)
    
    print("\n" + "="*50)
    run_eda(df)
    
    print("\n" + "="*50)
    df_processed = preprocess_dataset(df, apply_stemming=True)
    save_processed_data(df_processed)
    
    X, y, vectorizer = prepare_features(df_processed, use_tfidf=True, max_features=10000)
    
    models, X_train, X_test, y_train, y_test = train_models(X, y)
    
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    save_models(models)
    
    print("\n" + "="*50)
    results_df = evaluate_models(models, X_test, y_test)
    
    print("\n" + "="*50)
    print("MODEL OPTIMIZATION")
    print("="*50)
    
    best_model, best_params, grid_search = tune_logistic_regression(X_train, y_train, cv_folds=5)
    save_tuned_model(best_model)
    tuned_accuracy, tuned_f1 = evaluate_tuned_model(best_model, X_test, y_test)
    
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON: BEFORE vs AFTER TUNING")
    print("="*50)
    
    lr_before = results_df[results_df['Model'] == 'Logistic Regression'].iloc[0]
    print(f"\n{'Metric':<15} {'Before':<12} {'After':<12} {'Change':<12}")
    print("-" * 55)
    print(f"{'Accuracy':<15} {lr_before['Accuracy']:.4f}     {tuned_accuracy:.4f}     {tuned_accuracy - lr_before['Accuracy']:.4f}")
    print(f"{'F1 Score':<15} {lr_before['F1 Score']:.4f}     {tuned_f1:.4f}     {tuned_f1 - lr_before['F1 Score']:.4f}")
    
    print("\n" + "="*50)
    print("🎉 PROJECT COMPLETE!")
    print("="*50)
    
    print(f"\n🏆 Best Model: Logistic Regression (Tuned)")
    print(f"   Test Accuracy: {tuned_accuracy*100:.2f}%")
    print(f"   Test F1 Score: {tuned_f1*100:.2f}%")
    print(f"   Best Parameters: {best_params}")
    
    print("\n📁 Check these folders:")
    print("   - models/        : Saved models")
    print("   - outputs/       : Visualizations and reports")
    print("   - data/processed/: Cleaned dataset")
    print("\n🔮 To use the interactive predictor:")
    print("   python main.py")
    print("   Then choose option 2")

if __name__ == "__main__":
    main()