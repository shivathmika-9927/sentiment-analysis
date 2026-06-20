# src/evaluate_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import os

def evaluate_models(models, X_test, y_test, save_dir="outputs/reports/"):
    """
    Evaluate all trained models and print metrics.
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    for name, model in models.items():
        print(f"\n{'='*30}")
        print(f"Evaluating: {name}")
        print('='*30)
        
        try:
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
            
            print(f"✅ Accuracy:  {accuracy:.4f}")
            print(f"✅ Precision: {precision:.4f}")
            print(f"✅ Recall:    {recall:.4f}")
            print(f"✅ F1 Score:  {f1:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
            
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix:")
            print(cm)
            
            plot_confusion_matrix(cm, name, save_dir)
            
        except Exception as e:
            print(f"⚠️  Error evaluating {name}: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Accuracy', ascending=False)
        results_path = os.path.join(save_dir, "model_comparison.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to {results_path}")
        plot_model_comparison(results_df, save_dir)
    
    return results_df

def plot_confusion_matrix(cm, model_name, save_dir):
    """Plot and save confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    safe_name = model_name.lower().replace(' ', '_')
    filepath = os.path.join(save_dir, f"confusion_matrix_{safe_name}.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {filepath}")

def plot_model_comparison(results_df, save_dir):
    """Plot bar chart comparing all models."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    models = results_df['Model'].tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = results_df[metric].values
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    filepath = os.path.join(save_dir, "model_comparison_chart.png")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Model comparison chart saved to {filepath}")

if __name__ == "__main__":
    print("This script is meant to be imported and used from main.py")