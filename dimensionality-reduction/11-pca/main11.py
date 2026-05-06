"""
Experiment 11: Principal Component Analysis (PCA)
Author: Rudraksh Mehta

This program demonstrates PCA using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def main():

    # ---------------------------------------------------
    # Generate Correlated Dataset
    # ---------------------------------------------------

    np.random.seed(99)

    uncorrelated_data = np.random.randn(
        2,
        150
    )

    rotation_matrix = np.array([
        [0.7, -0.3],
        [0.8, 0.9]
    ])

    correlated_data = np.dot(
        rotation_matrix,
        uncorrelated_data
    ).T

    # ---------------------------------------------------
    # Apply PCA Transformation
    # ---------------------------------------------------

    pca_transformer = PCA(
        n_components=2
    )

    data_transformed = pca_transformer.fit_transform(
        correlated_data
    )

    # ---------------------------------------------------
    # Display Results
    # ---------------------------------------------------

    print("\n----- PCA Results -----")

    print(
        f"Original data shape: "
        f"{correlated_data.shape}"
    )

    print(
        f"Transformed data shape: "
        f"{data_transformed.shape}"
    )

    print(
        "Explained variance ratio per component:"
    )

    print(
        pca_transformer.explained_variance_ratio_
    )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    plt.style.use('default')

    plt.figure(figsize=(10, 5))

    # Original data plot
    plt.subplot(1, 2, 1)

    plt.scatter(
        correlated_data[:, 0],
        correlated_data[:, 1],
        alpha=0.7,
        c='purple'
    )

    plt.title('Original Data')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.axis('equal')

    # PCA transformed data plot
    plt.subplot(1, 2, 2)

    plt.scatter(
        data_transformed[:, 0],
        data_transformed[:, 1],
        alpha=0.7,
        c='orange'
    )

    plt.title('PCA Transformed Data')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.axis('equal')

    plt.suptitle('PCA Transformation Example')

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