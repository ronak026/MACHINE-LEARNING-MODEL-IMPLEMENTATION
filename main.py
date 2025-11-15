"""
Main script for spam email detection.

This script orchestrates the complete machine learning pipeline:
1. Load and preprocess data
2. Train multiple models
3. Evaluate models
4. Visualize results
5. Test with new emails
"""

import warnings
warnings.filterwarnings('ignore')

from data_loader import load_sample_data, preprocess_data, get_data_info
from model_trainer import train_models
from model_evaluator import (
    evaluate_models,
    get_best_model,
    get_classification_report,
    predict_email
)
from visualizer import (
    setup_plot_style,
    plot_label_distribution,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_feature_importance
)


def main():
    """Main execution function."""
    print("="*80)
    print("Spam Email Detection - Machine Learning Pipeline")
    print("="*80)
    print()
    
    # Setup visualization style
    setup_plot_style()
    
    # 1. Load data
    print("Step 1: Loading data...")
    df = load_sample_data()
    print("✓ Data loaded successfully!\n")
    
    # Display data info
    print("Data Overview:")
    print("-" * 80)
    print(f"Dataset shape: {df.shape}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print()
    
    # Visualize label distribution
    plot_label_distribution(df)
    
    # 2. Preprocess data
    print("\nStep 2: Preprocessing data...")
    X_train, X_test, y_train, y_test, vectorizer, feature_names = preprocess_data(df)
    print("✓ Data preprocessed successfully!\n")
    
    # Display preprocessing info
    print("Preprocessing Information:")
    print("-" * 80)
    get_data_info(df, X_train, X_test, y_train, y_test)
    print()
    
    # 3. Train models
    print("\nStep 3: Training models...")
    print("-" * 80)
    trained_models = train_models(X_train, y_train)
    print()
    
    # 4. Evaluate models
    print("\nStep 4: Evaluating models...")
    print("-" * 80)
    results_df, predictions_dict = evaluate_models(trained_models, X_test, y_test)
    print()
    
    # Visualize model comparison
    plot_model_comparison(results_df)
    
    # Plot confusion matrices
    plot_confusion_matrices(trained_models, X_test, y_test)
    
    # 5. Get best model
    print("\nStep 5: Analyzing best model...")
    print("-" * 80)
    best_model_name, best_model = get_best_model(trained_models, results_df)
    print(f"Best Model: {best_model_name}")
    print()
    
    # Detailed classification report
    print("Detailed Classification Report:")
    print("-" * 80)
    report = get_classification_report(best_model, X_test, y_test)
    print(report)
    print()
    
    # Feature importance
    print("\nStep 6: Feature importance analysis...")
    print("-" * 80)
    top_features = plot_feature_importance(best_model, feature_names)
    print()
    
    # 6. Test with new emails
    print("\nStep 7: Testing with new emails...")
    print("-" * 80)
    new_emails = [
        "Congratulations! You've won a free vacation. Click here to claim!",
        "Hi John, can we meet tomorrow to discuss the project?",
        "URGENT: Your account has been compromised. Verify immediately!",
        "Thanks for the update. I'll review the document and get back to you.",
        "Get rich quick! Earn $5000 per week from home. No experience needed!",
        "The meeting is scheduled for 2 PM in the conference room.",
    ]
    
    print(f"Testing {best_model_name} with new emails:\n")
    for email in new_emails:
        result = predict_email(best_model, vectorizer, email)
        print(f"Email: {result['email']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities - Ham: {result['probabilities']['ham']:.2%}, "
              f"Spam: {result['probabilities']['spam']:.2%}")
        print("-" * 80)
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

