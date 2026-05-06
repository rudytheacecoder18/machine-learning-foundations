"""
Experiment 15: Confusion Matrix
Author: Rudraksh Mehta

This program demonstrates Confusion Matrix
evaluation using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)


def main():

    # ---------------------------------------------------
    # Load Breast Cancer Dataset
    # ---------------------------------------------------

    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    # ---------------------------------------------------
    # Split Dataset
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # ---------------------------------------------------
    # Train Logistic Regression Model
    # ---------------------------------------------------

    model = LogisticRegression(
        max_iter=5000
    )

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------
    # Make Predictions
    # ---------------------------------------------------

    predictions = model.predict(X_test)

    # ---------------------------------------------------
    # Evaluate Model Performance
    # ---------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    confusion_mat = confusion_matrix(
        y_test,
        predictions
    )

    print("\n----- Confusion Matrix Results -----")

    print(
        f"\nAccuracy Score: "
        f"{accuracy:.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_mat)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names
        )
    )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat,
        display_labels=dataset.target_names
    ).plot(
        cmap='Blues',
        ax=ax
    )

    plt.title(
        'Confusion Matrix Visualization'
    )

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