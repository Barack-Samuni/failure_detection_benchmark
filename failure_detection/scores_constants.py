# Define the min/max values for each method explicitly (dict inside a dict)
score_ranges = {
    "Baseline": {"min": 0, "max": 1},
    "doctor_alpha": {"min": -1, "max": 0},
    "mcmc_soft": {"min": 0, "max": 1},
    "Laplace": {"min": 0, "max": 1},
    "ConfidNet": {"min": 0, "max": 1},
    "swag": {"min": 0, "max": 1}
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
    return confidence_level.get("Confidence",  {"min": None, "max": None})

def get_confidence_max():
    return confidence_level.get( "Confidence",{"max": None})
