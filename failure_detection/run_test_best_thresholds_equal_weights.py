import pandas as pd
import confidence_score as cs
import scores_constants
from pathlib import Path
import rejection_plots as pltr

test_df_path = Path(r"C:\Users\97254\Desktop\niv\AFEKA\M.sc Machine learning\AI 2th year\Computer vision\failure_detection_benchmark-main\github_rep_failure_detection\test_seed\seed_15\failure_detection\scores_df.csv")


test_df = pd.read_csv(test_df_path)
pd.set_option("display.max_rows", None)  # Show all rows

thresholds = cs.load_best_thresholds("best_thresholds.csv")

print("\n")
print(thresholds)


confidence_levels = list(range(0, 101, 10))  # List of confidence levels to test
rejection_df = cs.run_rejection_for_multiple_confidences(confidence_levels, test_df, thresholds, scores_constants.equal_weights, mode_debug=False)


# Create the DataFrame of predictions and rejections
result_df = cs.create_prediction_rejection_df(test_df, rejection_df, scores_constants.accuracy_rejection_columns_for_df)

accuracy_rejection_csv_path = "prediction_rejection_results.csv"

# Save to CSV
result_df.to_csv(accuracy_rejection_csv_path, index=False)  # index=False to exclude row indices


df = cs.load_data(accuracy_rejection_csv_path)
rejection_rates, accuracies = cs.compute_accuracy_vs_rejection(df)


#pltr.plot_rejection_vs_confidence_level(rejection_rates, confidence_levels)

cs.calculate_accuracy_vs_rejection_auc(rejection_rates, accuracies)
pltr.plot_accuracy_vs_rejection(rejection_rates, accuracies)
pltr.plot_accuracy_table(accuracies, rejection_rates, title="Accuracy vs. Rejection Rate Table")
confusion_matrices = pltr.compute_and_plot_rejector_confusion_matrix(df, rejection_rates)
