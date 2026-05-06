"""
Experiment 6: Gradient Boosting
Author: Rudraksh Mehta

This program demonstrates Gradient Boosting
Classification using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def main():

    # ---------------------------------------------------
    # Load Iris Dataset
    # ---------------------------------------------------

    iris = load_iris()

    X = iris.data
    y = iris.target

    # ---------------------------------------------------
    # Split Dataset
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=10
    )

    # ---------------------------------------------------
    # Create and Train Gradient Boosting Model
    # ---------------------------------------------------

    grad_boost_model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=2,
        random_state=10
    )

    grad_boost_model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Make Predictions
    # ---------------------------------------------------

    predictions = grad_boost_model.predict(X_test)

    # ---------------------------------------------------
    # Evaluate Model Performance
    # ---------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    print("\n----- Gradient Boosting Results -----")
    print(
        f"Accuracy of the Gradient Boosting Classifier: "
        f"{accuracy:.4f}"
    )

    # ---------------------------------------------------
    # Generate Confusion Matrix
    # ---------------------------------------------------

    confusion_mat = confusion_matrix(
        y_test,
        predictions
    )

    print("\n--- Confusion Matrix ---")
    print(confusion_mat)

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat,
        display_labels=iris.target_names
    ).plot(
        cmap='Oranges',
        ax=ax
    )

    plt.title("Gradient Boosting Confusion Matrix")

    # ---------------------------------------------------
    # Save Output Image
    # ---------------------------------------------------

    current_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    output_path = os.path.join(
        current_directory,
        "output.png"
    )

    plt.savefig(output_path)

    # Display graph
    plt.show()


# ---------------------------------------------------
# Run Program
# ---------------------------------------------------

if __name__ == "__main__":
    main()