"""
Experiment 10: Hierarchical Clustering
Author: Rudraksh Mehta

This program demonstrates Hierarchical Agglomerative
Clustering using SciPy and Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)

from sklearn.datasets import make_blobs


def main():

    # ---------------------------------------------------
    # Generate Synthetic Dataset
    # ---------------------------------------------------

    X, y = make_blobs(
        n_samples=15,
        n_features=2,
        centers=3,
        random_state=42
    )

    # ---------------------------------------------------
    # Perform Hierarchical Clustering
    # ---------------------------------------------------

    linked = linkage(
        X,
        method='average'
    )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    plt.figure(figsize=(10, 7))

    dendrogram(
        linked,
        orientation='right',
        labels=[f'P{i}' for i in range(len(X))],
        distance_sort='descending',
        show_leaf_counts=True
    )

    plt.title(
        "Hierarchical Clustering Dendrogram "
        "(Average Linkage)"
    )

    plt.xlabel("Euclidean Distance")
    plt.ylabel("Data Point")

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