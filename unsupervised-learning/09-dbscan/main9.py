"""
Experiment 9: DBSCAN Clustering
Author: Rudraksh Mehta

This program demonstrates DBSCAN Clustering
using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs


def main():

    # ---------------------------------------------------
    # Generate Synthetic Dataset
    # ---------------------------------------------------

    centers = [[1, 1], [-1, -1], [1, -1]]

    X, labels_true = make_blobs(
        n_samples=750,
        centers=centers,
        cluster_std=0.4,
        random_state=0
    )

    # ---------------------------------------------------
    # Apply DBSCAN Clustering
    # ---------------------------------------------------

    dbscan = DBSCAN(
        eps=0.3,
        min_samples=10
    )

    dbscan.fit(X)

    # ---------------------------------------------------
    # Extract Core Samples and Labels
    # ---------------------------------------------------

    core_samples_mask = np.zeros_like(
        dbscan.labels_,
        dtype=bool
    )

    core_samples_mask[
        dbscan.core_sample_indices_
    ] = True

    labels = dbscan.labels_

    # ---------------------------------------------------
    # Calculate Cluster Statistics
    # ---------------------------------------------------

    n_clusters_ = len(set(labels)) - (
        1 if -1 in labels else 0
    )

    n_noise_ = list(labels).count(-1)

    print("\n----- DBSCAN Clustering Results -----")
    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    plt.figure(figsize=(8, 6))

    unique_labels = set(labels)

    colors = [
        plt.cm.Spectral(each)
        for each in np.linspace(
            0,
            1,
            len(unique_labels)
        )
    ]

    for k, col in zip(unique_labels, colors):

        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        # Plot core samples
        xy = X[
            class_member_mask &
            core_samples_mask
        ]

        plt.plot(
            xy[:, 0],
            xy[:, 1],
            'o',
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=10
        )

        # Plot non-core samples
        xy = X[
            class_member_mask &
            ~core_samples_mask
        ]

        plt.plot(
            xy[:, 0],
            xy[:, 1],
            'o',
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=5
        )

    plt.title(
        f'DBSCAN Clustering | Clusters Found: {n_clusters_}'
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