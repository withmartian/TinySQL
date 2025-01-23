from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


def train_linear_probe(data, representation_column="representation", label_column="label", top_k=5):
    """
    Train a linear probe on a list of dictionaries with specified 'label' and 'representation' keys,
    return accuracy, and the top k most important features.

    Args:
        data (list): List of dictionaries with 'representation' and 'label' keys.
        representation_column (str): Key for the representation feature dictionary.
        label_column (str): Key for the label.
        top_k (int): Number of top features to return based on importance.

    Returns:
        accuracy (float): Accuracy of the linear probe on the test set.
        top_features (list): List of the top k most important features and their coefficients.
    """

    # Get the union of all feature keys across all dictionaries
    all_feature_keys = set()
    for item in data:
        all_feature_keys.update(item[representation_column].keys())
    all_feature_keys = sorted(all_feature_keys)  # Keep a consistent order of keys

    # Extract features and labels
    X = np.array([
        [item[representation_column].get(key, 0) for key in all_feature_keys]
        for item in data
    ])
    y = np.array([item[label_column] for item in data])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model (linear probe)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Get feature importance (absolute value of coefficients for interpretability)
    if len(model.coef_) == 1:  # Binary classification
        importance = np.abs(model.coef_[0])
    else:  # Multiclass classification
        importance = np.mean(np.abs(model.coef_), axis=0)

    # Get the top k features
    top_indices = importance.argsort()[-top_k:][::-1]
    top_features = [(all_feature_keys[i], importance[i]) for i in top_indices]

    return accuracy, top_features