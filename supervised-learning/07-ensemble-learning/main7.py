"""
Experiment 7: Ensemble Learning
Author: Rudraksh Mehta

This program demonstrates Ensemble Learning
using Voting Classifier in Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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
        test_size=0.3,
        random_state=99
    )

    # ---------------------------------------------------
    # Create Individual Models
    # ---------------------------------------------------

    clf1 = DecisionTreeClassifier(
        max_depth=4,
        random_state=99
    )

    clf2 = KNeighborsClassifier(
        n_neighbors=7
    )

    clf3 = GaussianNB()

    # ---------------------------------------------------
    # Create Ensemble Voting Classifier
    # ---------------------------------------------------

    ensemble_model = VotingClassifier(
        estimators=[
            ('dt', clf1),
            ('knn', clf2),
            ('gnb', clf3)
        ],
        voting='hard'
    )

    # ---------------------------------------------------
    # Train Model
    # ---------------------------------------------------

    ensemble_model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Make Predictions
    # ---------------------------------------------------

    predictions = ensemble_model.predict(X_test)

    # ---------------------------------------------------
    # Evaluate Model Performance
    # ---------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    print("\n----- Ensemble Learning Results -----")
    print(f"Hard-Voting Ensemble Accuracy: {accuracy:.4f}")

    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=wine.target_names
        )
    )

    # ---------------------------------------------------
    # Generate Confusion Matrix
    # ---------------------------------------------------

    confusion_mat = confusion_matrix(
        y_test,
        predictions
    )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat,
        display_labels=wine.target_names
    ).plot(
        cmap='Greens',
        ax=ax
    )

    plt.title("Ensemble Learning Confusion Matrix")

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