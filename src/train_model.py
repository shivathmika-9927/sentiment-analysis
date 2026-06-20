# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import time

def train_models(X, y, test_size=0.2, random_state=42):
    """
    Train multiple models and return them.
    """
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    # Split data
    print(f"\nSplitting data: {X.shape[0]} total samples")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    print(f"✅ Class balance in training: {np.bincount(y_train)}")
    
    models = {}
    training_times = {}
    
    # --- 1. Naive Bayes ---
    print("\n" + "-"*30)
    print("Training Naive Bayes...")
    start_time = time.time()
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    training_times['Naive Bayes'] = time.time() - start_time
    models['Naive Bayes'] = nb
    print(f"✅ Naive Bayes trained in {training_times['Naive Bayes']:.2f} seconds")
    
    # --- 2. Logistic Regression ---
    print("\n" + "-"*30)
    print("Training Logistic Regression...")
    start_time = time.time()
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=random_state,
        solver='liblinear'
    )
    lr.fit(X_train, y_train)
    training_times['Logistic Regression'] = time.time() - start_time
    models['Logistic Regression'] = lr
    print(f"✅ Logistic Regression trained in {training_times['Logistic Regression']:.2f} seconds")
    
    # --- 3. Random Forest ---
    print("\n" + "-"*30)
    print("Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    training_times['Random Forest'] = time.time() - start_time
    models['Random Forest'] = rf
    print(f"✅ Random Forest trained in {training_times['Random Forest']:.2f} seconds")
    
    # --- 4. SVM ---
    print("\n" + "-"*30)
    print("Training SVM (LinearSVC)...")
    start_time = time.time()
    svm = LinearSVC(
        max_iter=1000,
        C=1.0,
        random_state=random_state,
        dual=True
    )
    svm.fit(X_train, y_train)
    training_times['SVM'] = time.time() - start_time
    models['SVM'] = svm
    print(f"✅ SVM trained in {training_times['SVM']:.2f} seconds")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for name, time_taken in training_times.items():
        print(f"{name}: {time_taken:.2f} seconds")
    
    return models, X_train, X_test, y_train, y_test

def save_models(models, save_dir="models/"):
    """Save all trained models to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        filepath = os.path.join(save_dir, filename)
        joblib.dump(model, filepath)
        print(f"✅ {name} saved to {filepath}")

if __name__ == "__main__":
    print("This script is meant to be imported and used from main.py")