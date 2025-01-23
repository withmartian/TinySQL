from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def train_linear_probe(dataset, representation_column="representation", label_column="label", top_k=5):
    """
    Train a linear probe on a dataset with specified 'label' and 'representation' columns, 
    return accuracy, and the top k most important features.

    Args:
        dataset: Hugging Face dataset.
        representation_column (str): Name of the column containing feature dictionaries.
        label_column (str): Name of the column containing labels.
        top_k (int): Number of top features to return based on importance.

    Returns:
        accuracy: Accuracy of the linear probe on the test set.
        top_features: List of the top k most important features and their coefficients.
    """
    # Extract features and labels
    features_list = [list(rep.keys()) for rep in dataset[representation_column]]
    if len(set(map(tuple, features_list))) > 1:
        raise ValueError("All dictionaries in the representation column must have the same keys.")

    feature_names = features_list[0]
    X = np.array([list(rep.values()) for rep in dataset[representation_column]])
    y = np.array(dataset[label_column])

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