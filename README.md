# Task 4 — Spam Email Detection using Machine Learning

A comprehensive machine learning project that implements multiple classification models using scikit-learn to detect spam emails. This project demonstrates the complete machine learning pipeline from data preprocessing to model evaluation and visualization.

## Features

- **Multiple Classification Models**: Implements and compares 4 different algorithms:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- **Text Preprocessing**: Uses TF-IDF vectorization to convert text to numerical features
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score, and confusion matrices
- **Visualizations**: Interactive charts and graphs for model comparison and feature importance
- **Real-world Testing**: Demonstrates predictions on new email examples

## Project Structure

- `spam_email_detection.ipynb`: **Jupyter notebook** - Main deliverable showcasing model implementation and evaluation
- `main.py`: Python script to run the complete pipeline
- `data_loader.py`: Data loading and preprocessing module
- `model_trainer.py`: Model training module
- `model_evaluator.py`: Model evaluation and metrics module
- `visualizer.py`: Visualization and plotting module
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Prerequisites

- Python 3.9+ recommended
- Jupyter Notebook or JupyterLab (for running the notebook)
- pip package manager

## Setup

1. Navigate to the `Task-4` directory:

```bash
cd Task-4
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Jupyter Notebook (Main Deliverable)

The task requires a Jupyter notebook showcasing the model's implementation and evaluation:

1. Launch Jupyter Notebook:

```bash
jupyter notebook
```

2. Open `spam_email_detection.ipynb` in the Jupyter interface

3. Run all cells sequentially (Cell → Run All) or execute cells one by one

The notebook includes:

- Data loading and exploration
- Text preprocessing with TF-IDF vectorization
- Model implementation (4 different classifiers)
- Model evaluation with multiple metrics
- Visualizations (confusion matrices, performance comparisons)
- Testing with new email examples
- Feature importance analysis

### Option 2: Run the Main Script

Run the complete pipeline using the modular Python script:

```bash
python main.py
```

This will execute the entire machine learning pipeline:

- Load and preprocess data
- Train multiple models
- Evaluate and compare models
- Generate visualizations
- Test with new emails

### Use Individual Modules

You can also import and use individual modules in your own scripts:

```python
from data_loader import load_sample_data, preprocess_data
from model_trainer import train_models
from model_evaluator import evaluate_models, predict_email
from visualizer import plot_model_comparison

# Load and preprocess data
df = load_sample_data()
X_train, X_test, y_train, y_test, vectorizer, _ = preprocess_data(df)

# Train models
trained_models = train_models(X_train, y_train)

# Evaluate
results_df, _ = evaluate_models(trained_models, X_test, y_test)

# Visualize
plot_model_comparison(results_df)
```

### Understanding the Results

The main script provides:

- **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score for each model
- **Confusion Matrices**: Visual representation of true/false positives and negatives
- **Model Comparison**: Side-by-side comparison of all models
- **Feature Importance**: Top words/features that indicate spam emails
- **Predictions**: Real-time predictions on new email examples

## Model Performance

The script trains and evaluates 4 different models:

- All models are evaluated on the same test set
- Performance metrics are compared across models
- The best model is automatically identified based on F1-Score
- Detailed classification reports are provided

## Customization

### Using Your Own Dataset

To use your own email dataset:

1. Prepare a CSV file with two columns:

   - `email`: The email text content
   - `label`: Either "spam" or "ham"

2. Modify `data_loader.py` to load from CSV:

```python
# In main.py or your script:
from data_loader import load_data_from_csv, preprocess_data

df = load_data_from_csv('your_spam_emails.csv')
X_train, X_test, y_train, y_test, vectorizer, _ = preprocess_data(df)
```

### Adjusting Model Parameters

You can modify model hyperparameters in `model_trainer.py`:

```python
models = {
    'Naive Bayes': MultinomialNB(alpha=1.0),  # Adjust alpha
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale')
}
```

### Changing Train/Test Split

Modify the `test_size` parameter:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y_encoded,
    test_size=0.3,  # 30% for testing instead of 20%
    random_state=42,
    stratify=y_encoded
)
```

## Pipeline Steps

1. **Data Loading and Exploration**: Load dataset and visualize distribution
2. **Data Preprocessing**: Text vectorization and train/test split
3. **Model Training**: Train 4 different classification models
4. **Model Evaluation**: Compare performance metrics
5. **Visualization**: Charts and graphs for analysis
6. **Testing**: Predictions on new email examples
7. **Feature Importance**: Identify key spam indicators

## Output

The script generates:

- Performance metrics table
- Model comparison visualizations
- Confusion matrices for each model
- Feature importance charts
- Predictions on test emails with confidence scores

## Notes & Troubleshooting

- **Memory Issues**: If you have a large dataset, consider reducing `max_features` in the TF-IDF vectorizer
- **Low Accuracy**: Try adjusting model hyperparameters or increasing training data
- **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
- **Module Not Found**: Make sure you're running from the Task-4 directory

## Extending the Project

- **Add More Models**: Include additional classifiers like Gradient Boosting or Neural Networks
- **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV for optimization
- **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
- **Real-time API**: Create a Flask/FastAPI endpoint for real-time spam detection
- **Deep Learning**: Implement LSTM or Transformer models for better text understanding

## Dataset

The notebook includes a sample dataset for demonstration. For production use:

- Use larger, real-world datasets (e.g., SpamAssassin public corpus)
- Ensure balanced classes (similar number of spam and ham emails)
- Regularly update the dataset with new examples

## License

MIT (or as applicable to your project).
