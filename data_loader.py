"""
Data loading and preprocessing module for spam email detection.

This module handles:
- Loading email datasets
- Text preprocessing with TF-IDF vectorization
- Train/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_sample_data():
    """
    Load sample email dataset for demonstration.

    Returns:
        pd.DataFrame: DataFrame with 'email' and 'label' columns
    """
    sample_emails = [
        # Spam emails
        (
            "WINNER!! You have won $1,000,000! Click here to claim your prize now!",
            "spam",
        ),
        (
            "URGENT: Your account will be suspended. Verify your details immediately.",
            "spam",
        ),
        ("Free money! No investment required. Get rich quick scheme.", "spam"),
        ("Congratulations! You've been selected for a free iPhone. Claim now!", "spam"),
        ("Limited time offer! Buy now and get 90% discount. Act fast!", "spam"),
        ("You have won a lottery! Claim your prize worth $500,000 today.", "spam"),
        ("Click here for amazing deals! Lowest prices guaranteed.", "spam"),
        ("Your payment failed. Update your credit card information now.", "spam"),
        ("Exclusive offer just for you! Don't miss this opportunity.", "spam"),
        ("Act now! Limited stock available. Order before it's too late.", "spam"),
        (
            "You've been pre-approved for a loan. Apply now with no credit check.",
            "spam",
        ),
        ("Free trial! Cancel anytime. Sign up now for premium access.", "spam"),
        ("Your package delivery failed. Click to reschedule delivery.", "spam"),
        (
            "Earn money from home! Work from home opportunity. No experience needed.",
            "spam",
        ),
        ("Special promotion! Buy one get one free. Limited time only.", "spam"),
        # Ham (non-spam) emails
        ("Hi, can we schedule a meeting for tomorrow afternoon?", "ham"),
        ("Thank you for your email. I'll get back to you soon.", "ham"),
        ("The project deadline has been extended to next Friday.", "ham"),
        ("Please find attached the report you requested.", "ham"),
        ("Let's discuss the quarterly results in our next team meeting.", "ham"),
        ("I'll be out of office next week. Please contact my assistant.", "ham"),
        ("The conference call is scheduled for 3 PM today.", "ham"),
        ("Could you please review the document and provide feedback?", "ham"),
        ("I've completed the analysis. Here are the key findings.", "ham"),
        ("Thanks for your help with the project. Much appreciated!", "ham"),
        ("The meeting has been rescheduled to next Monday at 10 AM.", "ham"),
        ("Please confirm your attendance for the workshop next week.", "ham"),
        ("I've updated the spreadsheet with the latest data.", "ham"),
        ("Let me know if you need any additional information.", "ham"),
        ("Great work on the presentation! The client was impressed.", "ham"),
    ]

    return pd.DataFrame(sample_emails, columns=["email", "label"])


def load_data_from_csv(filepath):
    """
    Load email data from CSV file.

    Args:
        filepath: Path to CSV file with 'email' and 'label' columns

    Returns:
        pd.DataFrame: DataFrame with email data
    """
    return pd.read_csv(filepath)


def preprocess_data(df, test_size=0.2, random_state=42, max_features=1000):
    """
    Preprocess email data: vectorize text and split into train/test sets.

    Args:
        df: DataFrame with 'email' and 'label' columns
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility
        max_features: Maximum number of features for TF-IDF (default: 1000)

    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer, feature_names)
    """
    # Separate features and labels
    X = df["email"]
    y = df["label"]

    # Encode labels: spam = 1, ham = 0
    y_encoded = (y == "spam").astype(int)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
    )

    # Transform emails to feature vectors
    X_vectorized = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    return X_train, X_test, y_train, y_test, vectorizer, feature_names


def get_data_info(df, X_train, X_test, y_train, y_test):
    """
    Print information about the dataset.

    Args:
        df: Original DataFrame
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test labels
    """
    print(f"Dataset shape: {df.shape}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"\nLabel distribution percentage:")
    print(df["label"].value_counts(normalize=True) * 100)
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"\nTraining set label distribution:")
    print(f"  Ham (0): {(y_train == 0).sum()}")
    print(f"  Spam (1): {(y_train == 1).sum()}")
    print(f"\nTest set label distribution:")
    print(f"  Ham (0): {(y_test == 0).sum()}")
    print(f"  Spam (1): {(y_test == 1).sum()}")
