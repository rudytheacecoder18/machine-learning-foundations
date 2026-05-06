"""
Experiment 16: Hypothesis Testing
Author: Rudraksh Mehta

This program demonstrates Hypothesis Testing
using a one-sample t-test with SciPy.
"""

# ---------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


def main():

    # ---------------------------------------------------
    # Generate Sample Dataset
    # ---------------------------------------------------

    np.random.seed(42)

    sample_data = np.random.normal(
        loc=52,
        scale=8,
        size=100
    )

    # Hypothesized population mean
    population_mean = 50

    # ---------------------------------------------------
    # Perform One-Sample T-Test
    # ---------------------------------------------------

    t_statistic, p_value = stats.ttest_1samp(
        sample_data,
        population_mean
    )

    # ---------------------------------------------------
    # Display Results
    # ---------------------------------------------------

    print("\n----- Hypothesis Testing Results -----")

    print(
        f"\nSample Mean: "
        f"{np.mean(sample_data):.4f}"
    )

    print(
        f"T-Statistic: "
        f"{t_statistic:.4f}"
    )

    print(
        f"P-Value: "
        f"{p_value:.4f}"
    )

    # ---------------------------------------------------
    # Hypothesis Decision
    # ---------------------------------------------------

    alpha = 0.05

    print("\nDecision:")

    if p_value < alpha:

        print(
            "Reject the Null Hypothesis "
            "(statistically significant)"
        )

    else:

        print(
            "Fail to Reject the Null Hypothesis "
            "(not statistically significant)"
        )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(8, 6))

    plt.hist(
        sample_data,
        bins=12,
        edgecolor='black',
        alpha=0.75
    )

    plt.axvline(
        np.mean(sample_data),
        color='red',
        linestyle='--',
        linewidth=2,
        label='Sample Mean'
    )

    plt.axvline(
        population_mean,
        color='blue',
        linestyle=':',
        linewidth=2,
        label='Population Mean'
    )

    plt.title(
        'Hypothesis Testing Distribution'
    )

    plt.xlabel('Sample Values')
    plt.ylabel('Frequency')

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