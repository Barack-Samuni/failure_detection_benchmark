
import scores_constants

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


# normalize_scores_min_max is not relevant function because it might give undesired results for different training sessions
# def normalize_scores_min_max(score_list):
#     # Get the min and max of the list
#     min_val = min(score_list)
#     max_val = max(score_list)
#
#     # Apply min-max normalization to the list
#     normalized_list = [(x - min_val) / (max_val - min_val) * 100 for x in score_list]
#
#     return normalized_list


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
    normalized_list = [(x - min_val) / (max_val - min_val) * 100 for x in score_list]

    return normalized_list


