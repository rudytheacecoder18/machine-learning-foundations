"""
Experiment 12: Linear Discriminant Analysis (LDA)
Author: Rudraksh Mehta

This program demonstrates Linear Discriminant Analysis
using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
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
        random_state=88
    )

    # ---------------------------------------------------
    # Apply LDA Transformation
    # ---------------------------------------------------

    lda_transformer = LinearDiscriminantAnalysis(
        n_components=2
    )

    X_train_lda = lda_transformer.fit_transform(
        X_train,
        y_train
    )

    X_test_lda = lda_transformer.transform(
        X_test
    )

    # ---------------------------------------------------
    # Display Dataset Information
    # ---------------------------------------------------

    print("\n----- LDA Results -----")

    print(
        f"Original number of features: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Reduced number of features after LDA: "
        f"{X_train_lda.shape[1]}"
    )

    # ---------------------------------------------------
    # Train KNN Classifier on Reduced Data
    # ---------------------------------------------------

    classifier = KNeighborsClassifier(
        n_neighbors=5
    )

    classifier.fit(
        X_train_lda,
        y_train
    )

    # ---------------------------------------------------
    # Make Predictions
    # ---------------------------------------------------

    y_pred = classifier.predict(
        X_test_lda
    )

    # ---------------------------------------------------
    # Evaluate Model Performance
    # ---------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nAccuracy on LDA-transformed data: "
        f"{accuracy:.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_train_lda[:, 0],
        X_train_lda[:, 1],
        c=y_train,
        cmap='viridis',
        edgecolor='k',
        s=70
    )

    plt.title(
        'LDA Projection of Wine Dataset'
    )

    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')

    # Create proper legend
    handles, _ = scatter.legend_elements()

    plt.legend(
        handles,
        list(wine.target_names),
        title="Classes"
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