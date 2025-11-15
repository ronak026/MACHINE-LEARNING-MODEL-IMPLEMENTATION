"""
Model evaluation module for spam email detection.

This module handles evaluating trained models using various metrics.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate multiple models and return results.
    
    Args:
        trained_models: Dictionary mapping model names to trained models
        X_test: Test feature matrix
        y_test: Test labels
        
    Returns:
        tuple: (results_df, predictions_dict)
            - results_df: DataFrame with evaluation metrics
            - predictions_dict: Dictionary mapping model names to predictions
    """
    results = []
    predictions_dict = {}
    
    for name, model in trained_models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        predictions_dict[name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"{name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print("="*80)
    print("Summary of All Models:")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df, predictions_dict


def get_best_model(trained_models, results_df):
    """
    Get the best performing model based on F1-Score.
    
    Args:
        trained_models: Dictionary mapping model names to trained models
        results_df: DataFrame with evaluation results
        
    Returns:
        tuple: (best_model_name, best_model)
    """
    best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    best_model = trained_models[best_model_name]
    return best_model_name, best_model


def get_confusion_matrices(trained_models, X_test, y_test):
    """
    Get confusion matrices for all models.
    
    Args:
        trained_models: Dictionary mapping model names to trained models
        X_test: Test feature matrix
        y_test: Test labels
        
    Returns:
        dict: Dictionary mapping model names to confusion matrices
    """
    confusion_matrices = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
    
    return confusion_matrices


def get_classification_report(model, X_test, y_test, target_names=None):
    """
    Get detailed classification report for a model.
    
    Args:
        model: Trained model
        X_test: Test feature matrix
        y_test: Test labels
        target_names: List of target class names (default: ['Ham', 'Spam'])
        
    Returns:
        str: Classification report
    """
    if target_names is None:
        target_names = ['Ham', 'Spam']
    
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=target_names)


def predict_email(model, vectorizer, email_text):
    """
    Predict if an email is spam or ham.
    
    Args:
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
        email_text: Email text to predict
        
    Returns:
        dict: Prediction results with label, confidence, and probabilities
    """
    # Vectorize the email
    email_vectorized = vectorizer.transform([email_text])
    
    # Make prediction
    prediction = model.predict(email_vectorized)[0]
    prediction_proba = model.predict_proba(email_vectorized)[0]
    
    label = "SPAM" if prediction == 1 else "HAM"
    confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
    
    return {
        'email': email_text,
        'prediction': label,
        'confidence': confidence,
        'probabilities': {
            'ham': prediction_proba[0],
            'spam': prediction_proba[1]
        }
    }

