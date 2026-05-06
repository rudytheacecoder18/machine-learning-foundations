"""
Experiment 1: Linear Regression
Author: Rudraksh Mehta

This program demonstrates Linear Regression using Scikit-learn.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():

    # ---------------------------------------------------
    # Dataset Generation
    # ---------------------------------------------------

    # Set random seed for reproducibility
    np.random.seed(101)

    # Generate random feature values
    features = 2 * np.random.rand(100, 1)

    # Generate target values with noise
    target = 4 + 3 * features + np.random.randn(100, 1)

    # ---------------------------------------------------
    # Split Dataset into Training and Testing Sets
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.3,
        random_state=101
    )

    # ---------------------------------------------------
    # Create and Train Linear Regression Model
    # ---------------------------------------------------

    linear_model = LinearRegression()

    linear_model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Make Predictions
    # ---------------------------------------------------

    y_predictions = linear_model.predict(X_test)

    # ---------------------------------------------------
    # Evaluate Model Performance
    # ---------------------------------------------------

    mse = mean_squared_error(y_test, y_predictions)

    print("\n----- Linear Regression Results -----")
    print(f"Mean Squared Error: {mse:.4f}")

    # ---------------------------------------------------
    # Data Visualization
    # ---------------------------------------------------

    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(8, 6))

    # Plot actual test data
    plt.scatter(
        X_test,
        y_test,
        color='darkgreen',
        marker='o',
        label='Actual Test Data'
    )

    # Plot regression line
    plt.plot(
        X_test,
        y_predictions,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Fitted Regression Line'
    )

    # Labels and title
    plt.xlabel('Feature (X)')
    plt.ylabel('Target (y)')

    plt.title(
        f'Linear Regression Fit (MSE: {mse:.4f})'
    )

    plt.legend()

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