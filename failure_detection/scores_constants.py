
##########################################
######## PROJECT SCORES DEFINITIONS ######
##########################################

# Define the min/max values for each method explicitly (dict inside a dict)
score_ranges = {
    "Baseline": {"min": 0, "max": 1},
    "doctor_alpha": {"min": -1, "max": 0},
    "mcmc_soft_scores": {"min": 0, "max": 1},
    "Laplace_score": {"min": 0, "max": 1},
    "ConfidNet_scores": {"min": 0, "max": 1},
    "SWAG_score": {"min": 0, "max": 1}
}

# confidence = 0 -> reject all
# confidence = 100 -> reject nothing
confidence_level = {
    "Confidence": {"min": 0, "max": 100}
}

# Function to get min/max for a given method
def get_score_range(method_name):
    return score_ranges.get(method_name, {"min": None, "max": None})

def get_confidence_range():
    return confidence_level["Confidence"]  # No need for .get()

def get_confidence_max():
    return confidence_level["Confidence"]["max"]  # Directly return max value

num_scoring_methods = len(score_ranges)
confidence_max = get_confidence_max()  # Now correctly returns 100


# Prediction for accuracy is taken only from baseline for simplicity!
accuracy_rejection_columns_for_df = [
    'Targets',
    'Predictions',
    'Baseline',
    'doctor_alpha',
    'mcmc_soft_scores',
     'Laplace_score',
    'ConfidNet_scores',
    'SWAG_score'
]

# accuracy_rejection_columns_for_df = [
#     'Targets', 'Predictions', 'Baseline', 'doctor_alpha', 'mcmc_soft_scores',
#     'mcmc_predictions', 'Laplace_score', 'Laplace_predictions',
#     'ConfidNet_scores', 'SWAG_score', 'SWAG_predictions'
# ]


########################################
############## WEIGHTS #################
########################################
# Assign equal weights to all methods
equal_weights = {
    method: round(confidence_max / num_scoring_methods, 2)
    for method in score_ranges.keys()
}


