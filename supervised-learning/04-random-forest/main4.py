"""
Experiment 4: Random Forest
Author: Rudraksh Mehta

This program demonstrates Random Forest Classification
using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


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
        test_size=0.25,
        random_state=55
    )

    # ---------------------------------------------------
    # Create and Train Random Forest Model
    # ---------------------------------------------------

    rf_classifier = RandomForestClassifier(
        n_estimators=50,
        max_features='sqrt',
        random_state=55
    )

    rf_classifier.fit(X_train, y_train)

    # ---------------------------------------------------
    # Make Predictions
    # ---------------------------------------------------

    y_pred = rf_classifier.predict(X_test)

    # ---------------------------------------------------
    # Evaluate Model Performance
    # ---------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n----- Random Forest Results -----")
    print(f"Random Forest Accuracy: {accuracy:.2f}")

    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=iris.target_names
        )
    )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=iris.target_names,
        cmap='Blues',
        ax=ax
    )

    plt.title("Random Forest Confusion Matrix")

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