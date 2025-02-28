import pandas as pd
import confidence_score  as cs
import thresholding
import scores_constants



scores_df_path = r"scores_df.csv"

scores_df = pd.read_csv(scores_df_path)
pd.set_option("display.max_rows", None)  # Show all rows

# Extracting columns into lists
# Target and prediction columns
#is_correct_list = scores_df['IsCorrect'].tolist()
target_labels = scores_df["Targets"].tolist()
baseline_predictions = scores_df["Predictions"].tolist()
mcmc_predictions = scores_df["mcmc_predictions"].tolist()
Laplace_predictions = scores_df["Laplace_predictions"].tolist()
SWAG_predictions = scores_df["SWAG_predictions"].tolist()


# Scores columns
baseline_list = scores_df['Baseline'].tolist()
doctor_alpha_list = scores_df['doctor_alpha'].tolist()
mcmc_soft_scores_list = scores_df['mcmc_soft_scores'].tolist()
#mcmc_entropy_scores_list = scores_df['mcmc_entropy_scores'].tolist()
laplace_score_list = scores_df['Laplace_score'].tolist()
#trustscore_list = scores_df['TrustScore'].tolist()
confidnet_scores_list = scores_df['ConfidNet_scores'].tolist()
swag_score_list = scores_df['SWAG_score'].tolist()



# Call the function for each list
# from the csv scores file, it looks like doctor alpha predictions accuracy is equal to the baseline
baseline_correct, baseline_accuracy = cs.calculate_true_predictions(target_labels, baseline_predictions)
mcmc_correct, mcmc_accuracy = cs.calculate_true_predictions(target_labels, mcmc_predictions)
Laplace_correct, Laplace_accuracy = cs.calculate_true_predictions(target_labels, Laplace_predictions)
SWAG_correct, SWAG_accuracy = cs.calculate_true_predictions(target_labels, SWAG_predictions)

# Print results
print(f"Baseline Correct Predictions: {baseline_correct} ({baseline_accuracy:.2f}%)")
print(f"MCMC Correct Predictions: {mcmc_correct} ({mcmc_accuracy:.2f}%)")
print(f"Laplace Correct Predictions: {Laplace_correct} ({Laplace_accuracy:.2f}%)")
print(f"SWAG Correct Predictions: {SWAG_correct} ({SWAG_accuracy:.2f}%)")
print("\n\n")


# # Print 5 samples of the normalized lists (optional)
# Example usage
normalized_scores_df = cs.normalize_scores_df(scores_df)
print("Normalized scores to ramge 0 - 100 dict")
print(normalized_scores_df[:5])

thresholds = thresholding.find_thresholds(scores_df, visualize=True, separate_classes=False,confidnet=True, swag=True, duq=False,ensemble=False, mcmc_entropy_scores = False, trust_score = False)

print("\n")
print(thresholds)

confidence_level = 20

########################################################
# Select only the first row (keeping it as a DataFrame)
# first_row_df = scores_df.iloc[[0]]  # Use double brackets to maintain DataFrame structure
# filtered_first_row = first_row_df[list(thresholds.keys())]
# print(filtered_first_row)

# Apply rejection function to first row only
# rejected_first_sample = cs.weighted_rejection_by_confidence(confidence_level, filtered_first_row, thresholds, scores_constants.equal_weights, mode_debug= True)

# print(rejected_first_sample)  #
#######################################################


# Apply rejection function to entire df
rejected_df = cs.weighted_rejection_by_confidence(confidence_level, scores_df[:10], thresholds, scores_constants.equal_weights, mode_debug= False)
print(rejected_df)


# # Assuming scores_df and rejection_df are already defined
# prediction_rejection_df = cs.create_prediction_rejection_df(scores_df[10:20], rejected_df, scores_constants.accuracy_rejection_columns_for_df)
# print(prediction_rejection_df)
#


# Example usage:
confidence_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # List of confidence levels to test
rejection_df = cs.run_rejection_for_multiple_confidences(confidence_levels, scores_df, thresholds, scores_constants.equal_weights, mode_debug=False)


# Create the DataFrame
result_df = cs.create_prediction_rejection_df(scores_df, rejection_df, scores_constants.accuracy_rejection_columns_for_df)

accuracy_rejection_csv_path = "prediction_rejection_results.csv"
# Save to CSV
result_df.to_csv(accuracy_rejection_csv_path, index=False)  # index=False to exclude row indices


df = cs.load_data(accuracy_rejection_csv_path)
rejection_rates, accuracies = cs.compute_accuracy_vs_rejection(df)
cs.plot_accuracy_vs_rejection(rejection_rates, accuracies)