"""
Experiment 13: Singular Value Decomposition (SVD)
Author: Rudraksh Mehta

This program demonstrates Singular Value Decomposition
using NumPy.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt


def main():

    # ---------------------------------------------------
    # Generate Matrix
    # ---------------------------------------------------

    np.random.seed(50)

    original_matrix = np.random.randint(
        0,
        50,
        size=(4, 5)
    )

    # ---------------------------------------------------
    # Apply Singular Value Decomposition
    # ---------------------------------------------------

    U, S_vector, Vt = np.linalg.svd(
        original_matrix,
        full_matrices=False
    )

    # ---------------------------------------------------
    # Reconstruct Matrix
    # ---------------------------------------------------

    reconstructed_matrix = (
        U @ np.diag(S_vector) @ Vt
    )

    # ---------------------------------------------------
    # Display Results
    # ---------------------------------------------------

    print("\n----- Singular Value Decomposition Results -----")

    print("\n--- Original Matrix ---")
    print(original_matrix)

    print("\n--- U Matrix ---")
    print(U)

    print("\n--- Singular Values (S) ---")
    print(S_vector)

    print("\n--- Vt Matrix ---")
    print(Vt)

    print("\n--- Reconstructed Matrix ---")
    print(reconstructed_matrix)

    print(
        "\nReconstruction Successful:",
        np.allclose(
            original_matrix,
            reconstructed_matrix
        )
    )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.bar(
        range(1, len(S_vector) + 1),
        S_vector
    )

    plt.title('Singular Values from SVD')

    plt.xlabel('Component Number')
    plt.ylabel('Singular Value')

    plt.xticks(
        range(1, len(S_vector) + 1)
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