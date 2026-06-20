# app.py - Place this in the ROOT folder (sentiment-analysis/app.py)
import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import preprocess_text

# ============= PATHS =============
MODEL_PATH = "models/logistic_regression_tuned.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Sentiment Analysis Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============= CUSTOM CSS =============
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .positive-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .negative-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============= LOAD FUNCTIONS =============
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_vectorizer():
    return joblib.load(VECTORIZER_PATH)

@st.cache_resource
def get_sample_reviews():
    return [
        {"title": "💖 Very Positive", "text": "This is absolutely amazing! I love everything about it. Fantastic work!"},
        {"title": "😊 Positive", "text": "I really enjoyed this. It was well done and I would recommend it."},
        {"title": "😐 Neutral", "text": "It was okay. Nothing special, but not terrible either. Average experience."},
        {"title": "😞 Negative", "text": "This was terrible. I wasted my time and money. Very disappointed."},
        {"title": "🤬 Very Negative", "text": "Absolutely awful! Worst thing I've ever seen. Completely useless."},
        {"title": "❓ Mixed", "text": "It started good but then became boring and predictable."},
    ]

# ============= PREDICTION FUNCTION =============
def predict_review(text, model, vectorizer):
    """Predict sentiment for a single text."""
    try:
        cleaned = preprocess_text(str(text), apply_stemming=True)
        X = vectorizer.transform([cleaned])
        proba = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = proba[1] if prediction == 1 else proba[0]
        
        return sentiment, confidence, cleaned
    except Exception as e:
        return "Error", 0.0, str(e)

# ============= WORD CLOUD =============
def generate_wordcloud(text):
    """Generate word cloud from text."""
    if not text or len(text.strip()) < 10:
        return None
    
    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        )
        wordcloud = wc.generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception:
        return None

# ============= BATCH PREDICTION =============
def batch_predict(df, model, vectorizer):
    """Predict sentiment for all reviews in DataFrame."""
    results = []
    progress_bar = st.progress(0)
    
    for i, row in df.iterrows():
        text = str(row['review'])
        sentiment, confidence, cleaned = predict_review(text, model, vectorizer)
        results.append({
            'review': text,
            'cleaned_review': cleaned,
            'sentiment': sentiment,
            'confidence': confidence,
            'word_count': len(text.split())
        })
        progress_bar.progress((i + 1) / len(df))
    
    return pd.DataFrame(results)

# ============= SIDEBAR =============
def render_sidebar():
    """Render sidebar with model info and sample texts."""
    st.sidebar.title("🧠 Model Info")
    
    st.sidebar.markdown("""
    **Model:** Logistic Regression (Tuned)  
    **Accuracy:** 89.71%  
    **F1 Score:** 89.85%  
    **Features:** TF-IDF + Bigrams (10,000)  
    **Training Data:** 49,582 reviews  
    """)
    
    st.sidebar.divider()
    st.sidebar.subheader("📝 Sample Texts")
    
    for sample in get_sample_reviews():
        if st.sidebar.button(sample["title"], use_container_width=True):
            st.session_state.input_text = sample["text"]
            st.rerun()
    
    st.sidebar.divider()
    st.sidebar.markdown("""
    **💡 Tips:**  
    - This model works on ANY text, not just movie reviews!  
    - Try product reviews, tweets, or social media posts.  
    - Longer texts with clear sentiment work best.  
    """)

# ============= MAIN =============
def main():
    # Load models
    try:
        model = load_model()
        vectorizer = load_vectorizer()
        st.success("✅ Models loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Model files not found! Please train the model first using: python main.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()
    
    # Render sidebar (no model parameter needed)
    render_sidebar()
    
    # Title
    st.title("🧠 Sentiment Analysis Studio")
    st.markdown("**Analyze sentiment in ANY text — reviews, tweets, feedback, or social media posts.**")
    st.markdown("---")
    
    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction", "📊 Analytics Dashboard"])
    
    # ============= TAB 1: SINGLE PREDICTION =============
    with tab1:
        st.subheader("🔮 Analyze Any Text")
        st.markdown("Enter any text — reviews, feedback, social media posts, or comments.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "📝 Enter your text:",
                value=st.session_state.input_text,
                height=150,
                placeholder="e.g., This product is amazing! I love it..."
            )
        
        with col2:
            st.write("")
            st.write("")
            analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        if analyze_clicked and user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence, cleaned = predict_review(user_input, model, vectorizer)
                
                # Store in session
                st.session_state.input_text = user_input
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if sentiment == "Positive":
                        st.markdown(f"""
                        <div class="positive-box">
                            <h2>✅ {sentiment}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="negative-box">
                            <h2>❌ {sentiment}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                with col3:
                    if confidence > 0.8:
                        level = "🟢 High"
                    elif confidence > 0.6:
                        level = "🟡 Medium"
                    else:
                        level = "🔴 Low"
                    st.metric("Confidence Level", level)
                
                # Progress bar
                st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                
                # Word Cloud
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("🔍 See Preprocessed Text", expanded=False):
                        st.code(cleaned, language="text")
                
                with col2:
                    with st.expander("☁️ Word Cloud", expanded=True):
                        wordcloud_fig = generate_wordcloud(cleaned)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
                        else:
                            st.info("Text too short for word cloud.")
        
        elif analyze_clicked and not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
    
    # ============= TAB 2: BATCH PREDICTION =============
    with tab2:
        st.subheader("📁 Batch Prediction")
        st.markdown("Upload a CSV file with a **`review`** column to analyze multiple texts at once.")
        
        uploaded_file = st.file_uploader("📤 Choose CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'review' not in df.columns:
                    st.error("❌ CSV must contain a 'review' column!")
                else:
                    st.success(f"✅ File loaded! {len(df)} texts found.")
                    
                    with st.expander("📋 Preview Data", expanded=True):
                        st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("🚀 Analyze All", type="primary", use_container_width=True):
                        with st.spinner(f"Analyzing {len(df)} texts..."):
                            results_df = batch_predict(df, model, vectorizer)
                            st.session_state.batch_results = results_df
                            
                            # Summary
                            st.markdown("---")
                            st.subheader("📊 Summary")
                            
                            positive_count = len(results_df[results_df['sentiment'] == 'Positive'])
                            negative_count = len(results_df[results_df['sentiment'] == 'Negative'])
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Texts", len(results_df))
                            col2.metric("Positive", f"{positive_count/len(results_df)*100:.1f}%", f"{positive_count} texts")
                            col3.metric("Negative", f"{negative_count/len(results_df)*100:.1f}%", f"{negative_count} texts")
                            col4.metric("Avg Confidence", f"{results_df['confidence'].mean()*100:.1f}%")
                            
                            # Results table
                            st.subheader("📋 Full Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="sentiment_results.csv",
                                mime="text/csv"
                            )
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # ============= TAB 3: ANALYTICS DASHBOARD =============
    with tab3:
        st.subheader("📊 Analytics Dashboard")
        st.markdown("Visualize sentiment patterns from your batch prediction results.")
        
        if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
            df = st.session_state.batch_results
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", len(df))
            col2.metric("Positive", f"{len(df[df['sentiment']=='Positive'])/len(df)*100:.1f}%")
            col3.metric("Negative", f"{len(df[df['sentiment']=='Negative'])/len(df)*100:.1f}%")
            col4.metric("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%")
            
            # Charts
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment Distribution
                fig = px.pie(
                    df,
                    names='sentiment',
                    title='Sentiment Distribution',
                    color='sentiment',
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                    hole=0.3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence Distribution
                fig = px.histogram(
                    df,
                    x='confidence',
                    nbins=20,
                    title='Confidence Score Distribution',
                    color='sentiment',
                    color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Word Count Distribution
            fig = px.box(
                df,
                x='sentiment',
                y='word_count',
                title='Word Count by Sentiment',
                color='sentiment',
                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            fig = px.scatter(
                df,
                x='word_count',
                y='confidence',
                color='sentiment',
                title='Confidence vs Word Count',
                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                hover_data=['review']
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("📊 No batch results found. Upload a CSV in the Batch Prediction tab first.")

# ============= RUN =============
if __name__ == "__main__":
    main()