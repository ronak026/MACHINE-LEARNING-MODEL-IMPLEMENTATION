"""
Visualization module for spam email detection.

This module handles creating visualizations for model evaluation and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def setup_plot_style():
    """Set up matplotlib and seaborn styles."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")


def plot_label_distribution(df, save_path=None):
    """
    Plot distribution of email labels.
    
    Args:
        df: DataFrame with 'label' column
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    sns.countplot(data=df, x='label', ax=axes[0])
    axes[0].set_title('Email Label Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Label', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    
    # Pie chart
    df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
    axes[1].set_title('Label Distribution (Percentage)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Plot comparison of model performance metrics.
    
    Args:
        results_df: DataFrame with model evaluation results
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(results_df['Model'], results_df[metric], color=colors[idx], alpha=0.8)
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrices(trained_models, X_test, y_test, save_path=None):
    """
    Plot confusion matrices for all models.
    
    Args:
        trained_models: Dictionary mapping model names to trained models
        X_test: Test feature matrix
        y_test: Test labels
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        axes[idx].set_title(f'{name} - Confusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=11)
        axes[idx].set_ylabel('Actual', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance for a model.
    
    Args:
        model: Trained model
        feature_names: Array of feature names
        top_n: Number of top features to display (default: 20)
        save_path: Optional path to save the figure
    """
    # Get feature importance (for models that support it)
    if hasattr(model, 'feature_importances_'):
        # Random Forest
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression or SVM
        feature_importance = np.abs(model.coef_[0])
    else:
        # Naive Bayes - use log probabilities
        feature_importance = np.abs(
            model.feature_log_prob_[1] - model.feature_log_prob_[0]
        )
    
    # Get top features
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    top_features = [(feature_names[i], feature_importance[i]) for i in top_indices]
    
    print(f"Top {top_n} Most Important Features for Spam Detection:")
    print("="*80)
    for feature, importance in top_features:
        print(f"{feature:30s} : {importance:.4f}")
    
    # Visualize top features
    fig, ax = plt.subplots(figsize=(10, 8))
    features, importances = zip(*top_features)
    y_pos = np.arange(len(features))
    
    ax.barh(y_pos, importances, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Features for Spam Detection', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return top_features

