"""
Experiment 14: Gradient Descent
Author: Rudraksh Mehta

This program demonstrates Gradient Descent
optimization for Linear Regression.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt


def main():

    # ---------------------------------------------------
    # Generate Dataset
    # ---------------------------------------------------

    np.random.seed(42)

    X = 2 * np.random.rand(100, 1)

    y = 4 + 3 * X + np.random.randn(100, 1)

    # Add bias term
    X_b = np.c_[
        np.ones((100, 1)),
        X
    ]

    # ---------------------------------------------------
    # Gradient Descent Parameters
    # ---------------------------------------------------

    learning_rate = 0.1
    iterations = 1000
    m = len(X_b)

    theta = np.random.randn(2, 1)

    cost_history = []

    # ---------------------------------------------------
    # Perform Gradient Descent
    # ---------------------------------------------------

    for iteration in range(iterations):

        gradients = (
            2 / m
        ) * X_b.T.dot(
            X_b.dot(theta) - y
        )

        theta = theta - learning_rate * gradients

        predictions = X_b.dot(theta)

        cost = (
            1 / m
        ) * np.sum(
            (predictions - y) ** 2
        )

        cost_history.append(cost)

    # ---------------------------------------------------
    # Display Results
    # ---------------------------------------------------

    print("\n----- Gradient Descent Results -----")

    print("\nFinal Parameters (Theta):")
    print(theta)

    print(
        f"\nFinal Cost: "
        f"{cost_history[-1]:.4f}"
    )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(iterations),
        cost_history,
        color='blue',
        linewidth=2
    )

    plt.title(
        'Gradient Descent Cost Reduction'
    )

    plt.xlabel('Iterations')
    plt.ylabel('Cost Function Value')

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