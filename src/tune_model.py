# src/tune_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import time

def tune_logistic_regression(X_train, y_train, cv_folds=5):
    """
    Perform GridSearchCV to tune Logistic Regression hyperparameters.
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING (Logistic Regression)")
    print("="*50)
    
    print(f"\nData shape: {X_train.shape}")
    print(f"Cross-validation folds: {cv_folds}")
    
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    print("\nParameter grid:")
    print(f"  C: {param_grid['C']}")
    print(f"  penalty: {param_grid['penalty']}")
    print(f"  solver: {param_grid['solver']}")
    
    base_lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    
    print(f"\nRunning GridSearchCV with {cv_folds}-fold cross-validation...")
    print(f"Total combinations to evaluate: {len(param_grid['C']) * len(param_grid['penalty'])}")
    print("This may take 2-3 minutes...")
    
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ GridSearchCV completed in {elapsed_time:.2f} seconds")
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("\n" + "="*50)
    print("TUNING RESULTS")
    print("="*50)
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation F1 score: {best_score:.4f}")
    
    print("\nAll combinations tested:")
    results_df = pd.DataFrame(grid_search.cv_results_)
    for i, row in results_df.iterrows():
        params = row['params']
        mean_score = row['mean_test_score']
        print(f"  {params} -> F1: {mean_score:.4f}")
    
    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='f1_macro')
    print(f"\nCross-validation F1 scores for best model: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return best_model, best_params, grid_search

def evaluate_tuned_model(model, X_test, y_test):
    """Evaluate the tuned model on the test set."""
    print("\n" + "="*50)
    print("EVALUATING TUNED MODEL ON TEST SET")
    print("="*50)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    print(f"✅ Test F1 Score: {f1:.4f}")
    
    return accuracy, f1

def save_tuned_model(model, save_path="models/logistic_regression_tuned.pkl"):
    """Save the tuned model to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"✅ Tuned model saved to {save_path}")

if __name__ == "__main__":
    print("This script is meant to be imported and used from main.py")