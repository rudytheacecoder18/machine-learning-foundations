"""
Experiment 3: Decision Trees
Author: Rudraksh Mehta

This program demonstrates Decision Tree Classification
using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


def main():

    # ---------------------------------------------------
    # Load Iris Dataset
    # ---------------------------------------------------

    iris_dataset = load_iris()

    X = iris_dataset.data
    y = iris_dataset.target

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
    # Create and Train Decision Tree Model
    # ---------------------------------------------------

    decision_tree_clf = DecisionTreeClassifier(
        max_depth=3,
        random_state=99
    )

    decision_tree_clf.fit(X_train, y_train)

    # ---------------------------------------------------
    # Make Predictions
    # ---------------------------------------------------

    predictions = decision_tree_clf.predict(X_test)

    # ---------------------------------------------------
    # Evaluate Model Performance
    # ---------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    print("\n----- Decision Tree Results -----")
    print(f"Model Accuracy: {accuracy:.4f}")

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    plt.figure(figsize=(15, 10))

    plot_tree(
        decision_tree_clf,
        feature_names=iris_dataset.feature_names,
        class_names=iris_dataset.target_names,
        filled=True,
        rounded=True,
        fontsize=10
    )

    plt.title(
        "Decision Tree for Iris Classification (Max Depth = 3)"
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