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
    # Extract features and labels
    features_list = [list(item[representation_column].keys()) for item in data]
    if len(set(map(tuple, features_list))) > 1:
        raise ValueError("All dictionaries in the representation column must have the same keys.")

    feature_names = features_list[0]
    X = np.array([list(item[representation_column].values()) for item in data])
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
    top_features = [(feature_names[i], importance[i]) for i in top_indices]

    return accuracy, top_features