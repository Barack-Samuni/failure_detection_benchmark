import pandas as pd
import confidence_score  as cs
import thresholding

scores_df_path = r"scores_df.csv"

scores_df = pd.read_csv(scores_df_path)

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

# Normalize each of the score lists
baseline_list_normalized = cs.normalize_scores_absolute_range("Baseline",baseline_list)
doctor_alpha_list_normalized = cs.normalize_scores_absolute_range("doctor_alpha",doctor_alpha_list)
mcmc_soft_scores_list_normalized = cs.normalize_scores_absolute_range("mcmc_soft",mcmc_soft_scores_list)
# mcmc_entropy_scores_list_normalized = cs.normalize_scores_absolute_range(mcmc_entropy_scores_list)
laplace_score_list_normalized = cs.normalize_scores_absolute_range("Laplace",laplace_score_list)
# trustscore_list_normalized = cs.normalize_scores_absolute_range(trustscore_list)
confidnet_scores_list_normalized = cs.normalize_scores_absolute_range("ConfidNet",confidnet_scores_list)
swag_score_list_normalized = cs.normalize_scores_absolute_range("swag",swag_score_list)

# Print 5 samples of the normalized lists (optional)
print(f"Baseline Normalized: {baseline_list_normalized[:5]}")  # Print first 5 as an example
print(f"Doctor Alpha Normalized: {doctor_alpha_list_normalized[:5]}")
print(f"MCMC Soft Scores Normalized: {mcmc_soft_scores_list_normalized[:5]}")
#print(f"MCMC Entropy Scores Normalized: {mcmc_entropy_scores_list_normalized[:5]}")
print(f"Laplace Score Normalized: {laplace_score_list_normalized[:5]}")
#print(f"TrustScore Normalized: {trustscore_list_normalized[:5]}")
print(f"ConfidNet Scores Normalized: {confidnet_scores_list_normalized[:5]}")
print(f"SWAG Score Normalized: {swag_score_list_normalized[:5]}")


# scoring_methods = ["Baseline", "doctor_alpha", "mcmc_soft_scores", "Laplace_score","ConfidNet_scores", "SWAG_score"]

thresholds = thresholding.find_thresholds(scores_df, visualize=True, separate_classes=False,confidnet=True, swag=True, duq=False,ensemble=False, mcmc_entropy_scores = False, trust_score = False)

print("\n")
print(thresholds)

