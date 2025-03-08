
import scores_constants
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import thresholding
import seaborn as sns

def calculate_true_predictions(target_labels, predictions_list):
    # Step 1: Convert TRUE/FALSE to 1/0 in predictions list
    predictions_list = [1 if pred else 0 for pred in predictions_list]

    # Step 2: Calculate the number of correct predictions
    correct_predictions = sum(
        [1 if target == prediction else 0 for target, prediction in zip(target_labels, predictions_list)])

    # Step 3: Calculate the percentage of correct predictions
    total_predictions = len(target_labels)  # Total number of predictions
    accuracy_percentage = (correct_predictions / total_predictions) * 100  # Percentage formula

    # Return the count of correct predictions and the accuracy percentage
    return correct_predictions, accuracy_percentage



def calculate_accuracy_of_each_score(scores_df):
    """
    Function only for checking performance of each score
    """
    # Extracting columns into lists
    # Target and prediction columns
    # is_correct_list = scores_df['IsCorrect'].tolist()
    target_labels = scores_df["Targets"].tolist()
    baseline_predictions = scores_df["Predictions"].tolist()
    mcmc_predictions = scores_df["mcmc_predictions"].tolist()
    Laplace_predictions = scores_df["Laplace_predictions"].tolist()
    SWAG_predictions = scores_df["SWAG_predictions"].tolist()

    # Scores columns
    baseline_list = scores_df['Baseline'].tolist()
    doctor_alpha_list = scores_df['doctor_alpha'].tolist()
    mcmc_soft_scores_list = scores_df['mcmc_soft_scores'].tolist()
    # mcmc_entropy_scores_list = scores_df['mcmc_entropy_scores'].tolist()
    laplace_score_list = scores_df['Laplace_score'].tolist()
    # trustscore_list = scores_df['TrustScore'].tolist()
    confidnet_scores_list = scores_df['ConfidNet_scores'].tolist()
    swag_score_list = scores_df['SWAG_score'].tolist()

    # Call the function for each list
    # from the csv scores file, it looks like doctor alpha predictions accuracy is equal to the baseline
    baseline_correct, baseline_accuracy = calculate_true_predictions(target_labels, baseline_predictions)
    mcmc_correct, mcmc_accuracy = calculate_true_predictions(target_labels, mcmc_predictions)
    Laplace_correct, Laplace_accuracy = calculate_true_predictions(target_labels, Laplace_predictions)
    SWAG_correct, SWAG_accuracy = calculate_true_predictions(target_labels, SWAG_predictions)

    # Print results
    print(f"Baseline Correct Predictions: {baseline_correct} ({baseline_accuracy:.2f}%)")
    print(f"MCMC Correct Predictions: {mcmc_correct} ({mcmc_accuracy:.2f}%)")
    print(f"Laplace Correct Predictions: {Laplace_correct} ({Laplace_accuracy:.2f}%)")
    print(f"SWAG Correct Predictions: {SWAG_correct} ({SWAG_accuracy:.2f}%)")
    print("\n\n")


"""
Function Name: normalize_scores_absolute_range
breif: The function takes as input the absolute possible min and max value of the score method
and returns the  normalized score list 
"""
def normalize_scores_absolute_range(method_name, score_list):
    # Get min/max for the given method
    range_values = scores_constants.get_score_range(method_name)
    min_val, max_val = range_values["min"], range_values["max"]

    # Ensure valid min/max values to prevent division by zero
    if min_val is None or max_val is None or min_val == max_val:
        raise ValueError(f"Invalid min/max values for method: {method_name}")

    # Apply min-max normalization
    normalized_list = [round((x - min_val) / (max_val - min_val), 2) * 100 for x in score_list]

    return normalized_list


def normalize_scores_df(scores_df):
    """
    Normalizes all columns in the scores DataFrame using the predefined method-specific min/max ranges.

    Args:
        scores_df (pd.DataFrame): DataFrame where each column corresponds to a method and contains scores.

    Returns:
        pd.DataFrame: DataFrame with normalized scores.
    """
    normalized_dict = {}

    for method_name in scores_df.columns:
        if method_name in scores_constants.score_ranges.keys():
            score_list = scores_df[method_name].tolist()
            normalized_dict[method_name] = normalize_scores_absolute_range(method_name, score_list)

    return pd.DataFrame(normalized_dict, index=scores_df.index)



def find_best_thresholds(weights:dict, train_root: str, val_root: str, output_csv: str = "best_thresholds.csv"):
    """Finds the best threshold set by evaluating multiple training and validation datasets."""
    train_root = Path(train_root)
    val_root = Path(val_root)

    print("Train directory exists:", train_root.exists())
    print("Validation directory exists:", val_root.exists())

    # Automatically gather training and validation files
    training_files = sorted(train_root.rglob("scores_df.csv"))
    validation_files = sorted(val_root.rglob("scores_df.csv"))

    # Debugging: Print the discovered files
    print("Training files found:", training_files)
    print("Validation files found:", validation_files)

    # Store thresholds and corresponding AUC scores
    threshold_sets = []
    auc_scores = []

    # Confidence levels (constant)
    confidence_levels = list(range(0, 101, 10))

    # Compute thresholds from training sets
    for train_file in training_files:
        train_df = pd.read_csv(train_file)
        thresholds = thresholding.find_thresholds(
            train_df, visualize=True, separate_classes=False,
            confidnet=True, swag=True, duq=False, ensemble=False,
            mcmc_entropy_scores=False, trust_score=False
        )
        print(thresholds)
        threshold_sets.append(thresholds)

    print(threshold_sets)
    print("\n")

    # Evaluate thresholds on validation sets
    for thresholds in threshold_sets:
        auc_list = []
        for val_file in validation_files:
            val_df = pd.read_csv(val_file)
            rejection_df = run_rejection_for_multiple_confidences(
                confidence_levels, val_df, thresholds, weights, mode_debug=False
            )
            result_df = create_prediction_rejection_df(
                val_df, rejection_df, scores_constants.accuracy_rejection_columns_for_df
            )

            rejection_rates, accuracies = compute_accuracy_vs_rejection(result_df)
            auc = calculate_accuracy_vs_rejection_auc(rejection_rates, accuracies)
            auc_list.append(auc)

        # Compute average AUC for this threshold set
        avg_auc = sum(auc_list) / len(auc_list)
        auc_scores.append(avg_auc)

    # Find the best threshold set (highest average AUC)
    best_index = auc_scores.index(max(auc_scores))
    best_thresholds = threshold_sets[best_index]
    best_auc = auc_scores[best_index]  # Get corresponding AUC


    # Save the best thresholds to CSV
    output_path = Path(output_csv)
    best_thresholds_df = pd.DataFrame([best_thresholds])
    best_thresholds_df.to_csv(output_path, index=False)

    print(f"Best thresholds saved to {output_path}\n")
    print(f"Best threshold set saved with AUC: {max(auc_scores)}")

    return best_thresholds, best_auc



def handle_edge_cases(confidence_level, confidence_range, scores_df, thresholds, weights):
    # Handle edge cases in confidence level
    if confidence_level < confidence_range["min"]:
        confidence_level = confidence_range["min"]

    if confidence_level > confidence_range["max"]:
        confidence_level = confidence_range["max"]

    # Handle empty data
    if scores_df.empty:
        raise ValueError("Scores DataFrame cannot be empty.")
    if not thresholds:
        raise ValueError("Thresholds dictionary cannot be empty.")
    if not weights:
        raise ValueError("Weights dictionary cannot be empty.")

    return confidence_level


def weighted_rejection_by_confidence(confidence_level, scores_df, thresholds, weights, mode_debug):
    """
    Rejects samples based on a weighted score contribution and confidence level.
    confidence_level means how much you trust your trained model.
    If confidence_level is high, it means the user trusts its model, and if it is low, the user does not trust it.
    The confidence_level is then converted to uncertainty_level, which is: uncertainty_level = 100-confidence_level

    If enough scores passed their thresholds and resulted in sum of weighted scores that passed the uncertainty_level,
    the sample will not be rejected.
    If not enough scores passed their thresholds, there will not be enough certainty for the classification of the sample
    and the sample will be rejected.

    Args:
        confidence_level (float): Confidence level of the user in the accuracy of the model (0-100).
        scores_df (pd.DataFrame): DataFrame where columns are method names and rows are score values.
        thresholds (dict): Dictionary where keys are method names and values are threshold values.
        weights (dict): Dictionary where keys are method names and values are the weight for passing the threshold.

    Returns:
        pd.Series: Boolean Series indicating rejection (True = reject, False = accept).
    """
    confidence_range = scores_constants.get_confidence_range()

    # Handle edge cases
    confidence_level = handle_edge_cases(confidence_level, confidence_range, scores_df, thresholds, weights)

    # If confidence level is 0, the model is not to be trusted, reject all. Do not even check  scores
    if confidence_level == confidence_range["min"]:
        return pd.Series([True] * len(scores_df), index=scores_df.index)  # Reject all

    # If confidence level is 100, the model is highly accurate, reject  nothing. Do not even check  scores
    if confidence_level == confidence_range["max"]:
        return pd.Series([False] * len(scores_df), index=scores_df.index)  # Accept all

    # Get the uncertainty_level of the model , what level of trust you give to your scores
    # Represents the confidence Threshold
    uncertainty_level = confidence_range["max"] - confidence_level

    # Filter columns that exist in both thresholds and weights
    valid_methods = set(thresholds.keys()) & set(weights.keys())
    filtered_scores_df = scores_df[list(valid_methods)]

    # Align the thresholds Series with the DataFrame columns to avoid system warnings
    filtered_scores_df, thresholds_series = filtered_scores_df.align(pd.Series(thresholds), axis=1, copy=False)

    # Determine which scores pass the threshold
    passed_score_threshold = filtered_scores_df >= pd.Series(thresholds_series)

    # Assign weights to passing scores
    weighted_scores = passed_score_threshold * pd.Series(weights)

    # Compute total accumulated weight per sample
    total_weight_per_sample = weighted_scores.sum(axis=1)

    # Compute acceptance threshold based on confidence level
    max_possible_weight = int(sum(weights.values()))
    acceptance_threshold = (uncertainty_level / confidence_range["max"]) * max_possible_weight  # Scale confidence to weight

    # Print mid-results if in mode debug
    if mode_debug == True:
        print(f" Passed scores threshold: {passed_score_threshold}")
        print(f" weighted_scores: {weighted_scores}")
        print(f" total_weight_per_sample: {total_weight_per_sample}")
        print(f" max_possible_weight: {max_possible_weight}")
        print(f" acceptance_threshold: {acceptance_threshold} \n")

    #  Reject if accumulated weight is below the acceptance threshold
    return total_weight_per_sample < acceptance_threshold



def run_rejection_for_multiple_confidences(confidence_levels, scores_df, thresholds, weights, mode_debug):
    """
    Runs the weighted_rejection_by_confidence function for multiple confidence levels
    and stores the rejection results in a DataFrame.

    Args:
        confidence_levels (list): List of confidence levels to evaluate.
        scores_df (pd.DataFrame): DataFrame with score values.
        thresholds (dict): Dictionary of threshold values for each method.
        weights (dict): Dictionary of weight values for each method.
        mode_debug (bool): Debug mode flag.

    Returns:
        pd.DataFrame: DataFrame where each column represents rejection results for a different confidence level.
    """
    rejection_results = {}

    for confidence_level in confidence_levels:
        rejection_results[f"Rejected_{confidence_level}"] = weighted_rejection_by_confidence(
            confidence_level, scores_df, thresholds, weights, mode_debug
        )

    rejection_df = pd.DataFrame(rejection_results, index=scores_df.index)

    return rejection_df



def create_prediction_rejection_df(scores_df, rejection_df, accuracy_rejection_columns_for_df):
    """
    Creates a new DataFrame containing the target columns from scores_df along with multiple rejection status columns.

    Args:
        scores_df (pd.DataFrame): DataFrame containing scores and predictions.
        rejection_df (pd.DataFrame): DataFrame containing rejection status for multiple confidence levels.
        accuracy_rejection_columns_for_df (list): List of column names to extract from scores_df.

    Returns:
        pd.DataFrame: A DataFrame with the necessary columns from scores_df and multiple "Rejected" columns.
    """

    # Ensure that all required columns exist in scores_df
    missing_columns = [col for col in accuracy_rejection_columns_for_df if col not in scores_df.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing from scores_df: {', '.join(missing_columns)}")

    # Extract the required columns from scores_df
    result_df = scores_df[accuracy_rejection_columns_for_df].copy()

    # Ensure rejection_df index aligns with scores_df before merging
    rejection_df = rejection_df.loc[result_df.index]

    # Add all rejection columns from rejection_df
    result_df = result_df.join(rejection_df)

    return result_df


def load_data(csv_path):
    """Loads the CSV file into a DataFrame."""
    return pd.read_csv(csv_path)

def compute_accuracy_vs_rejection(df):
    """
    Computes model accuracy at different rejection thresholds.

    Args:
        df (pd.DataFrame): The DataFrame containing targets, predictions, and rejection columns.

    Returns:
        list: Rejection rates (percentages).
        list: Corresponding model accuracies.
    """
    targets = df["Targets"]
    predictions = df["Predictions"]

    rejection_columns = [col for col in df.columns if "Rejected_" in col]
    rejection_rates = []
    accuracies = []

    for col in rejection_columns:
        mask = df[col]  # True means rejected
        accepted_targets = targets[~mask]  # Keep only non-rejected samples
        accepted_predictions = predictions[~mask]

        if len(accepted_targets) > 0:
            accuracy = (accepted_targets == accepted_predictions).mean()
        else:
            accuracy = 0  # Set accuracy to 0 when all samples are rejected
            #accuracy = None  # Creates a better looking graph but might be not good

        rejection_rate = mask.mean() * 100  # Convert to percentage
        rejection_rates.append(rejection_rate)
        accuracies.append(accuracy)

    return rejection_rates, accuracies



def calculate_accuracy_vs_rejection_auc(rejection_rates, accuracies):
    # Ensure data is sorted in ascending order
    sorted_indices = np.argsort(rejection_rates)  # Get sorting indices
    rejection_rates = np.array(rejection_rates)[sorted_indices]  # Sort rejection rates
    accuracies = np.array(accuracies)[sorted_indices]  # Sort accuracies accordingly

    # Compute area under the curve using the trapezoidal rule
    auc = np.trapz(accuracies, rejection_rates)
    print(f"Area under the curve (AUC): {auc:.4f}")

    return auc


def load_best_thresholds(csv_path: str) -> dict:
    """Load the best thresholds from a CSV file and return as a dictionary."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("The CSV file is empty.")

    # Convert the first row to a dictionary
    return df.iloc[0].to_dict()


