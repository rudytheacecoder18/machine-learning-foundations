"""
Experiment 2: Logistic Regression
Author: Rudraksh Mehta

This program demonstrates Logistic Regression using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():

    # ---------------------------------------------------
    # Generate Synthetic Classification Dataset
    # ---------------------------------------------------

    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=101
    )

    # ---------------------------------------------------
    # Split Dataset
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=101
    )

    # ---------------------------------------------------
    # Create and Train Logistic Regression Model
    # ---------------------------------------------------

    log_reg_model = LogisticRegression(C=0.5)

    log_reg_model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Make Predictions
    # ---------------------------------------------------

    y_pred = log_reg_model.predict(X_test)

    # ---------------------------------------------------
    # Evaluate Model Performance
    # ---------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n----- Logistic Regression Results -----")
    print(f"Model Accuracy: {accuracy:.2f}")

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap='coolwarm',
        edgecolors='k',
        s=60
    )

    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    # Predict values on mesh grid
    Z = log_reg_model.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(
        xx,
        yy,
        Z,
        cmap='Pastel1',
        alpha=0.8
    )

    # Labels and title
    plt.title(
        'Logistic Regression Decision Boundary (C=0.5)'
    )

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

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