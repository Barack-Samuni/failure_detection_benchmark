import pandas as pd
import confidence_score as cs
import thresholding
import scores_constants
from pathlib import Path


# Root paths for training and validation datasets
train_root = Path(r"C:\Users\97254\Desktop\niv\AFEKA\M.sc Machine learning\AI 2th year\Computer vision\failure_detection_benchmark-main\github_rep_failure_detection\train_seeds")
val_root = Path(r"C:\Users\97254\Desktop\niv\AFEKA\M.sc Machine learning\AI 2th year\Computer vision\failure_detection_benchmark-main\github_rep_failure_detection\val_seeds")

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
        rejection_df = cs.run_rejection_for_multiple_confidences(
            confidence_levels, val_df, thresholds, scores_constants.equal_weights, mode_debug=False
        )
        result_df = cs.create_prediction_rejection_df(
            val_df, rejection_df, scores_constants.accuracy_rejection_columns_for_df
        )

        rejection_rates, accuracies = cs.compute_accuracy_vs_rejection(result_df)
        auc = cs.calculate_accuracy_vs_rejection_auc(rejection_rates, accuracies)
        auc_list.append(auc)

    # Compute average AUC for this threshold set
    avg_auc = sum(auc_list) / len(auc_list)
    auc_scores.append(avg_auc)

# Find the best threshold set (highest average AUC)
best_index = auc_scores.index(max(auc_scores))
best_thresholds = threshold_sets[best_index]

# Save the best thresholds to CSV
output_path = Path("best_thresholds.csv")
best_thresholds_df = pd.DataFrame([best_thresholds])
best_thresholds_df.to_csv(output_path, index=False)
print(f"Best thresholds: saved to {best_thresholds_df}\n")
print(f"Best threshold set saved to {output_path} with AUC:", max(auc_scores))