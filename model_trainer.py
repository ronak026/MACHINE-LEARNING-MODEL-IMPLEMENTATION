"""
Model training module for spam email detection.

This module handles training multiple classification models.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_models():
    """
    Get dictionary of classification models to train.
    
    Returns:
        dict: Dictionary mapping model names to model instances
    """
    return {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42, probability=True)
    }


def train_models(X_train, y_train, models=None):
    """
    Train multiple classification models.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        models: Dictionary of models to train (if None, uses default models)
        
    Returns:
        dict: Dictionary mapping model names to trained model instances
    """
    if models is None:
        models = get_models()
    
    trained_models = {}
    print("Training models...\n")
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} trained successfully!\n")
    
    print("All models trained successfully!")
    return trained_models

