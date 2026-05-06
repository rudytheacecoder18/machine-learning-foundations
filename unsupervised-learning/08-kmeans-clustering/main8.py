"""
Experiment 8: K-Means Clustering
Author: Rudraksh Mehta

This program demonstrates K-Means Clustering
using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def main():

    # ---------------------------------------------------
    # Generate Synthetic Dataset
    # ---------------------------------------------------

    features, true_labels = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=0.80,
        random_state=101
    )

    # ---------------------------------------------------
    # Elbow Method to Find Optimal Clusters
    # ---------------------------------------------------

    inertias = []

    k_range = range(1, 11)

    for k in k_range:

        kmeans = KMeans(
            n_clusters=k,
            n_init='auto',
            random_state=101
        )

        kmeans.fit(features)

        inertias.append(kmeans.inertia_)

    # ---------------------------------------------------
    # Plot Elbow Method
    # ---------------------------------------------------

    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(8, 6))

    plt.plot(
        k_range,
        inertias,
        marker='o',
        linestyle='--',
        color='purple'
    )

    plt.title('Elbow Method for Optimal k')

    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')

    plt.xticks(list(k_range))

    # ---------------------------------------------------
    # Save Elbow Method Graph
    # ---------------------------------------------------

    current_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    elbow_output_path = os.path.join(
        current_directory,
        "elbow_method.png"
    )

    plt.savefig(elbow_output_path)

    plt.show()

    # ---------------------------------------------------
    # Apply Final K-Means Model
    # ---------------------------------------------------

    final_kmeans = KMeans(
        n_clusters=4,
        n_init='auto',
        random_state=101
    )

    final_kmeans.fit(features)

    predicted_labels = final_kmeans.labels_

    centers = final_kmeans.cluster_centers_

    # ---------------------------------------------------
    # Visualize Clustering Result
    # ---------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        features[:, 0],
        features[:, 1],
        c=predicted_labels,
        s=50,
        cmap='viridis',
        label='Data Points'
    )

    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c='red',
        s=200,
        alpha=0.75,
        marker='X',
        label='Centroids'
    )

    plt.title('K-Means Clustering Result (k=4)')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.legend()

    # ---------------------------------------------------
    # Save Final Clustering Graph
    # ---------------------------------------------------

    final_output_path = os.path.join(
        current_directory,
        "output.png"
    )

    plt.savefig(final_output_path)

    # Display graph
    plt.show()


# ---------------------------------------------------
# Run Program
# ---------------------------------------------------

if __name__ == "__main__":
    main()