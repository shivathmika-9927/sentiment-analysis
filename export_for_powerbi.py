# export_for_powerbi.py
import pandas as pd
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import preprocess_text

def export_predictions_for_powerbi():
    """
    Export predictions for all reviews to a CSV file for Power BI.
    """
    print("="*50)
    print("📊 EXPORTING DATA FOR POWER BI")
    print("="*50)
    
    # Load model and vectorizer
    model_path = "models/logistic_regression_tuned.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please run full pipeline first.")
        print("   Run: python main.py")
        return
    
    # Load data
    data_path = "data/processed/cleaned_data.csv"
    if not os.path.exists(data_path):
        print("❌ Cleaned data not found! Run the full pipeline first.")
        print("   Run: python main.py")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} reviews")
    
    # Load model
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model and vectorizer loaded")
    
    # Make predictions on all reviews
    print("\n🔮 Making predictions on all reviews...")
    
    sentiments = []
    confidences = []
    
    for idx, row in df.iterrows():
        text = str(row['cleaned_review'])
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        sentiments.append(sentiment)
        
        # Confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_tfidf)[0]
            confidence = proba[1] if prediction == 1 else proba[0]
        else:
            decision = model.decision_function(text_tfidf)[0]
            confidence = 1 / (1 + pd.np.exp(-decision))
            if prediction == 0:
                confidence = 1 - confidence
        confidences.append(confidence)
    
    # Add predictions to dataframe
    df['predicted_sentiment'] = sentiments
    df['confidence'] = confidences
    df['review_length'] = df['review'].str.len()
    df['word_count'] = df['review'].str.split().str.len()
    
    # Add sentiment flags for easier counting in Power BI
    df['is_positive'] = (df['predicted_sentiment'] == 'Positive').astype(int)
    df['is_negative'] = (df['predicted_sentiment'] == 'Negative').astype(int)
    df['is_actual_positive'] = (df['sentiment'] == 'positive').astype(int)
    df['is_correct'] = (df['sentiment'] == df['predicted_sentiment'].str.lower()).astype(int)
    
    # Select columns for Power BI (keep only needed ones)
    powerbi_df = df[[
        'review', 
        'sentiment', 
        'predicted_sentiment', 
        'confidence', 
        'review_length',
        'word_count',
        'is_positive',
        'is_negative',
        'is_actual_positive',
        'is_correct'
    ]]
    
    # Save to CSV
    output_path = "dashboard/powerbi_data.csv"
    os.makedirs("dashboard", exist_ok=True)
    powerbi_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Data exported to {output_path}")
    print(f"📊 Total records: {len(powerbi_df)}")
    print(f"   Actual Positive: {len(powerbi_df[powerbi_df['sentiment'] == 'positive'])}")
    print(f"   Actual Negative: {len(powerbi_df[powerbi_df['sentiment'] == 'negative'])}")
    print(f"   Predicted Positive: {len(powerbi_df[powerbi_df['predicted_sentiment'] == 'Positive'])}")
    print(f"   Predicted Negative: {len(powerbi_df[powerbi_df['predicted_sentiment'] == 'Negative'])}")
    print(f"   Correct Predictions: {powerbi_df['is_correct'].sum()}")
    print(f"   Accuracy: {powerbi_df['is_correct'].sum() / len(powerbi_df) * 100:.2f}%")
    
    print("\n📁 File saved at: dashboard/powerbi_data.csv")
    print("   Open this file in Power BI to create your dashboard.")

if __name__ == "__main__":
    export_predictions_for_powerbi()