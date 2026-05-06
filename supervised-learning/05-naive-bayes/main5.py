"""
Experiment 5: Naive Bayes
Author: Rudraksh Mehta

This program demonstrates Gaussian Naive Bayes
Classification using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def main():

    # ---------------------------------------------------
    # Load Wine Dataset
    # ---------------------------------------------------

    wine = load_wine()

    X = wine.data
    y = wine.target

    # ---------------------------------------------------
    # Split Dataset
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=101
    )

    # ---------------------------------------------------
    # Create and Train Gaussian Naive Bayes Model
    # ---------------------------------------------------

    naive_bayes_model = GaussianNB()

    naive_bayes_model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Make Predictions
    # ---------------------------------------------------

    y_pred = naive_bayes_model.predict(X_test)

    # ---------------------------------------------------
    # Evaluate Model Performance
    # ---------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n----- Naive Bayes Results -----")
    print(f"Model Accuracy: {accuracy:.2%}")

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=wine.target_names
        )
    )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=wine.target_names,
        cmap='Purples',
        ax=ax
    )

    plt.title("Naive Bayes Confusion Matrix")

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