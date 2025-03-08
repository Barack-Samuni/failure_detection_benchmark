import scores_constants
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import thresholding
import seaborn as sns
from confidence_score import calculate_accuracy_vs_rejection_auc

def plot_accuracy_vs_rejection(rejection_rates, accuracies):
    """Plots the accuracy vs. rejection rate."""
    plt.figure(figsize=(8, 5))

    area_under_curve = calculate_accuracy_vs_rejection_auc(rejection_rates, accuracies)

    # # scale the area under curve to bumber between 0 and 1
    area_under_curve = area_under_curve / 100
    # # Filter out None values
    valid_accuracies = [a for a in accuracies if a is not None]

    # Present the accuracies as percentages
    valid_accuracies = [acc * 100 for acc in valid_accuracies]  # Convert to [0,100]

    valid_rejection_rates = [r for r, a in zip(rejection_rates, valid_accuracies) if a is not None]  # Keep only valid pairs

    if not valid_accuracies:  # Check if there's still valid data left
        print("Error: No valid accuracy values to plot.")
        return

    plt.plot(valid_rejection_rates, valid_accuracies, marker="o", linestyle="-", label="Accuracy vs. Rejection Rate")

    plt.xlabel("Rejection Rate (%)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Model Accuracy as a Function of Rejection Rate\nArea under curve: {round(area_under_curve, 1)} ")
    plt.legend()

    # Set finer grid steps
    plt.xticks(np.arange(0, 101, 5))  # X-axis: 10% steps
    plt.yticks(np.arange(0, 101, 5))  # Y-axis: 5 steps in accuracy

    plt.grid()

    plt.show()


def plot_rejection_vs_confidence_level(rejection_rates, confidence_levels):
    """Plots the accuracy vs. rejection rate."""
    plt.figure(figsize=(8, 5))

    # Filter out None values

    valid_rejection_rates = [r for r, a in zip(rejection_rates, confidence_levels) if a is not None]  # Keep only valid pairs


    plt.plot(confidence_levels, valid_rejection_rates , marker="o", linestyle="-", label="Rejection Rate vs. Confidence Level ")

    plt.xlim(min(valid_rejection_rates), max(valid_rejection_rates))
    plt.ylim(min(confidence_levels), max(confidence_levels))

    plt.xlabel("Confidence level")
    plt.ylabel("Rejection Rate (%)")
    plt.title("Rejection Rate as function of confidence level")
    plt.legend()

    # # Set finer grid steps
    plt.xticks(np.arange(0, 101, 10))  # X-axis: 10% steps
    plt.yticks(np.arange(0, 101, 10))  # Y-axis: 0.1 steps in accuracy

    plt.grid()

    plt.show()

def plot_accuracy_table(accuracies, rejection_rates, title):
    """
    Plots a table displaying accuracies and rejection rates.

    Parameters:
    accuracies (list): List of corresponding accuracies.
    rejection_rates (list): List of rejection rates (%).

    Returns:
    None (Displays the table)
    """
    # Round values for better readability
    formatted_accuracies = [f"{acc:.2f}" for acc in accuracies]
    formatted_rejection_rates = [f"{rej:.1f}" for rej in rejection_rates]

    # Create figure and axis for the table
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("tight")  # Remove extra white space
    ax.axis("off")  # Hide axes

    # Add title
    plt.title(title, fontsize=12, fontweight="bold", pad=10)

    # Normal order (high to low accuracy) .Combine lists into table format (flipping order)
    #table_data = list(zip(formatted_accuracies, formatted_rejection_rates))

    # Reverse the order of the data from low accuracy to high accuracy
    table_data = list(reversed(list(zip(formatted_accuracies, formatted_rejection_rates))))

    # Create the table
    table = ax.table(cellText=table_data, colLabels=["Accuracy", "Rejection Rate (%)"],
                     cellLoc="center", loc="center")

    # Display the table
    plt.show()


def compute_and_plot_rejector_confusion_matrix(df, rejection_rates):
    """
    Computes a confusion matrix for each confidence level, plots it, and returns the confusion matrices.

    Parameters:
    df (DataFrame): DataFrame containing targets, predictions, and rejection columns.
    rejection_rates (list): List of rejection rates corresponding to confidence levels.

    Returns:
    dict: A dictionary where keys are confidence levels and values are confusion matrices.
    """
    # Copy DataFrame to avoid modifying the original
    df_copy = df.copy()
    formatted_rejection_rates = [f"{rej:.1f}" for rej in rejection_rates]

    # Convert "TRUE"/"FALSE" to boolean
    df_copy["Predictions"] = df_copy["Predictions"].astype(bool)

    confidence_levels = [int(col.split("_")[-1]) for col in df_copy.columns if col.startswith("Rejected_")]
    confusion_matrices = {}

    for conf_level in confidence_levels:
        reject_col = f"Rejected_{conf_level}"

        # Define True/False Positives/Negatives based on rejection and correctness
        correct_predictions = df_copy["Targets"] == df_copy["Predictions"]
        df_copy[reject_col] = df_copy[reject_col].astype(bool)
        rejected = df_copy[reject_col]

        tp_reject = np.sum(correct_predictions & rejected)  # Correct predictions & rejected
        fp_reject = np.sum(~correct_predictions & rejected)  # Wrong predictions & rejected
        tn_accept = np.sum(correct_predictions & ~rejected)  # Correct predictions & not rejected
        fn_accept = np.sum(~correct_predictions & ~rejected)  # Wrong predictions & not rejected

        # Store results
        cm_total = tp_reject + fp_reject + tn_accept + fn_accept
        cm_percent = np.array([
            [tp_reject / cm_total * 100 if cm_total else 0, tn_accept / cm_total * 100 if cm_total else 0],
            [fp_reject / cm_total * 100 if cm_total else 0, fn_accept / cm_total * 100 if cm_total else 0]
        ])
        confusion_matrices[conf_level] = np.array([
            [tp_reject, tn_accept],
            [fp_reject, fn_accept]
        ])

        # Plot confusion matrices side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Confidence Level {conf_level} (Rejection Rate: {formatted_rejection_rates[confidence_levels.index(conf_level)]}%)",
            fontsize=14, fontweight="bold")

        sns.heatmap(confusion_matrices[conf_level], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Rejected", "Not Rejected"],
                    yticklabels=["Correct Prediction", "Wrong Prediction"], ax=axes[0])
        axes[0].set_title("Confusion Matrix (Counts)")
        axes[0].set_xlabel("Rejection Decision")
        axes[0].set_ylabel("Prediction Correctness")

        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=["Rejected", "Not Rejected"],
                    yticklabels=["Correct Prediction", "Wrong Prediction"], ax=axes[1])
        axes[1].set_title("Confusion Matrix (Percentages)")
        axes[1].set_xlabel("Rejection Decision")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit suptitle
        plt.show()

    return confusion_matrices
