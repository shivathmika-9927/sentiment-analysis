# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def plot_sentiment_distribution(df: pd.DataFrame, save_path: str = "outputs/visualizations/"):
    """
    Plot bar chart and pie chart of sentiment distribution.
    
    Args:
        df: DataFrame with 'sentiment' column
        save_path: Directory to save plots
    """
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Bar Chart ---
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#66b3b2', '#ff9999']
    bars = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors, edgecolor='black')
    ax1.set_title('Sentiment Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    # --- Pie Chart ---
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, explode=(0.05, 0.05), shadow=True)
    ax2.set_title('Sentiment Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Save the combined figure
    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{save_path}sentiment_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sentiment distribution plot saved to {save_path}sentiment_distribution.png")


def plot_text_length_distribution(df: pd.DataFrame, save_path: str = "outputs/visualizations/"):
    """
    Plot histogram of text lengths by sentiment.
    
    Args:
        df: DataFrame with 'review' and 'sentiment' columns
        save_path: Directory to save plots
    """
    # Calculate text length (number of characters)
    df['text_length'] = df['review'].str.len()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram for positive and negative reviews
    for sentiment, color in zip(['positive', 'negative'], ['#66b3b2', '#ff9999']):
        subset = df[df['sentiment'] == sentiment]
        ax.hist(subset['text_length'], bins=50, alpha=0.6, label=sentiment.capitalize(),
                color=color, edgecolor='black', linewidth=0.5)
    
    ax.set_title('Text Length Distribution by Sentiment', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Characters', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{save_path}text_length_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Text length distribution plot saved to {save_path}text_length_distribution.png")


def generate_wordclouds(df: pd.DataFrame, save_path: str = "outputs/visualizations/"):
    """
    Generate word clouds for positive and negative reviews.
    
    Args:
        df: DataFrame with 'review' and 'sentiment' columns
        save_path: Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Separate reviews by sentiment
    positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['review'].astype(str))
    negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['review'].astype(str))
    
    # Generate word clouds
    wordcloud_positive = WordCloud(width=800, height=400,
                                   background_color='white',
                                   colormap='Greens',
                                   max_words=100,
                                   contour_width=1,
                                   contour_color='black').generate(positive_reviews)
    
    wordcloud_negative = WordCloud(width=800, height=400,
                                   background_color='white',
                                   colormap='Reds',
                                   max_words=100,
                                   contour_width=1,
                                   contour_color='black').generate(negative_reviews)
    
    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1.imshow(wordcloud_positive, interpolation='bilinear')
    ax1.set_title('Positive Reviews Word Cloud', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(wordcloud_negative, interpolation='bilinear')
    ax2.set_title('Negative Reviews Word Cloud', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}wordclouds.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Word clouds saved to {save_path}wordclouds.png")


def run_eda(df: pd.DataFrame):
    """
    Run all EDA functions.
    
    Args:
        df: DataFrame with 'review' and 'sentiment' columns
    """
    print("\n=== Generating EDA Visualizations ===\n")
    
    # 1. Sentiment distribution
    plot_sentiment_distribution(df)
    
    # 2. Text length distribution
    plot_text_length_distribution(df)
    
    # 3. Word clouds
    generate_wordclouds(df)
    
    print("\n✅ EDA completed! Check outputs/visualizations/ for all plots.")


if __name__ == "__main__":
    # Quick test
    df = pd.read_csv("data/raw/IMDB Dataset.csv")
    run_eda(df)