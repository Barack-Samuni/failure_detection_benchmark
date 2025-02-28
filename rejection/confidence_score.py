import pandas as pd
import numpy as np

def find_optimal_weights(dataframe, thresholds):
    """
    This function finds the optimal weights for the weighted sum used to calculate the confidence score
    First, it finds where to assign the rejection region: whether above or below the threshold according
    to the percentage of correct classifications in each region - the region with the higher percentage will
    be the acceptance region.
    Then, in order to find the weights, for each rejection region for each score - the function calculates the
    percentage of rejected samples over the overall incorrect classifications made by this score.
    After calculating for all scores - they will be normalized by 1 - dividing their distance from the maximum value
    gained by the maximum value gained. This method assumes no class separation!

    :param dataframe: pandas dataframe containing the scoring methods and their scores.
    :param thresholds: pandas dataframe with a scoring method column and the value of the threshold for that method.
    :return: pandas dataframe with the columns: Scoring_method, Threshold, Direction, Weight.
    """
    output_columns = ["Scoring_Method", "Threshold", "Rejection_Direction", "Weight"]
    optimal_weights = pd.DataFrame(columns=output_columns)
    optimal_weights["Scoring_Method"] = thresholds["Scoring_Method"]
    optimal_weights["Threshold"] = thresholds["Threshold"]

    weights = np.array([])

    for scoring_method in optimal_weights["Scoring_Method"]:
        threshold = optimal_weights.loc[optimal_weights["Scoring_Method"] == scoring_method, "Threshold"].values[0]

        # find correct and incorrect classifications for that scoring method
        if scoring_method == "mcmc_soft_scores" or scoring_method == "mcmc_entropy_scores":
            correct = dataframe[dataframe['mcmc_predictions'] == dataframe['Targets']][scoring_method]
            incorrect = dataframe[dataframe['mcmc_predictions'] != dataframe['Targets']][scoring_method]

        elif scoring_method == "Laplace_score":
            correct = dataframe[dataframe['Laplace_predictions'] == dataframe['Laplace_targets']][scoring_method]
            incorrect = dataframe[dataframe['Laplace_predictions'] != dataframe['Laplace_targets']][scoring_method]

        elif scoring_method == "SWAG_score":
            correct = dataframe[dataframe['SWAG_targets'] == dataframe['SWAG_predictions']][scoring_method]
            incorrect = dataframe[dataframe['SWAG_targets'] != dataframe['SWAG_predictions']][scoring_method]

        else:
            correct = dataframe[dataframe['IsCorrect'] == True][scoring_method]
            incorrect = dataframe[dataframe['IsCorrect'] == False][scoring_method]

        # find the percentage of correct classifications for each region for that scoring method
        correct_above_threshold_percentage = (correct[correct >= threshold].shape[0] / correct.shape[0]) * 100
        correct_below_threshold_percentage = (correct[correct < threshold].shape[0] / correct.shape[0]) * 100

        if correct_above_threshold_percentage >= correct_below_threshold_percentage:  # reject below
            optimal_weights.loc[optimal_weights["Scoring_Method"] == scoring_method, "Rejection_Direction"] = "<"

        else:                                                                        # reject above
            optimal_weights.loc[optimal_weights["Scoring_Method"] == scoring_method, "Rejection_Direction"] = ">"


        # Now find the percentage of incorrect classifications in the rejection area - so you van give this score a weight
        if optimal_weights.loc[optimal_weights["Scoring_Method"] == scoring_method, "Rejection_Direction"].iloc[0] == "<":
           rejection_of_incorrect_percentage = (incorrect[incorrect < threshold].shape[0] / incorrect.shape[0]) * 100

        else:
            rejection_of_incorrect_percentage = (incorrect[incorrect >= threshold].shape[0] / incorrect.shape[0]) * 100

        weights = np.append(weights, rejection_of_incorrect_percentage)

    max_weight = weights.max()
    weights = 1 - (np.abs(weights - max_weight) / max_weight)
    optimal_weights["Weight"] = weights

    return optimal_weights

