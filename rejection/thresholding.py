import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def find_thresholds(dataframe,visualize=True, separate_classes=False,confidnet=False, swag=False, duq=False,ensemble=False):
    scoring_methods = ["Baseline", "doctor_alpha", "mcmc_soft_scores", "mcmc_entropy_scores", "Laplace_score", "TrustScore"]
    # Extract unique classes from the dataframe
    classes = sorted(dataframe["Targets"].unique())

    if confidnet:
        scoring_methods.append("ConfidNet_scores")
    if swag:
        scoring_methods.append("SWAG_score")
    if duq:
        scoring_methods.append("DUQ_score")
    if ensemble:
        scoring_methods.append("Ensemble_score")

    thresholds = {}

    if visualize:
        if separate_classes:
            fig, axes = plt.subplots(nrows=len(scoring_methods), ncols=len(classes), figsize=(20, 36))  # Increased figsize
            fig.suptitle('Histograms for Correct and Incorrect Classifications', fontsize=20, y=0.98)

        else:
            fig, axes = plt.subplots(nrows=len(scoring_methods), ncols=1, figsize=(8, 36))

            # Add suptitle above the subplots
            fig.suptitle('Histograms for Correct and Incorrect Classifications (No Class Separation)', fontsize=20, y=0.98)


        fig.subplots_adjust(top=0.93,hspace=0.5)  # Adjust top to create space for the suptitle
        # Add suptitle above the subplots
        plt.subplots_adjust(top=0.93, hspace=0.5)  # Adjust overall top margin and add blank space with hspace

    else:
        axes = None


    if separate_classes:
        for row, method in enumerate(scoring_methods):
            # doctor alpha and not already normalized
            if method == "doctor_alpha" and dataframe["doctor_alpha"].min() >= -1 and dataframe["doctor_alpha"].max() <= 0:
                dataframe['doctor_alpha'] = normalize_doctor_alpha(dataframe)

            for col, cls in enumerate(classes):
                if visualize:
                    ax = axes[row, col]

                else:
                    ax = None

                if swag and method == "SWAG_score":
                    correct = dataframe[(dataframe['SWAG_targets'] == cls) &
                                        (dataframe['SWAG_predictions'] == dataframe['SWAG_targets'])][method]
                    incorrect = dataframe[(dataframe['SWAG_targets'] == cls) &
                                        (dataframe['SWAG_predictions'] != dataframe['SWAG_targets'])][method]

                elif method == "mcmc_soft_scores" or method == "mcmc_entropy_scores":
                    correct = dataframe[(dataframe['Targets'] == cls) &
                                        (dataframe['mcmc_predictions'] == dataframe['Targets'])][method]

                    incorrect = dataframe[(dataframe['Targets'] == cls) &
                                        (dataframe['mcmc_predictions'] != dataframe['Targets'])][method]

                elif method == "Laplace_score":
                    correct = dataframe[(dataframe['Targets'] == cls) &
                                        (dataframe['Laplace_predictions'] == dataframe['Laplace_targets'])][method]
                    incorrect = dataframe[(dataframe['Targets'] == cls) &
                                        (dataframe['Laplace_predictions'] != dataframe['Laplace_targets'])][method]

                else:
                    # Separate the data into correct and incorrect classifications for the current class and scoring method
                    correct = dataframe[(dataframe['Targets'] == cls) & (dataframe['IsCorrect'] == True)][method]
                    incorrect = dataframe[(dataframe['Targets'] == cls) & (dataframe['IsCorrect'] == False)][method]

                # Find the optimal threshold using KDE without clearing histograms
                x_vals = np.linspace(min(dataframe[method]), max(dataframe[method]), 1000)

                if len(correct) >= 2:
                    correct_kde = sns.kdeplot(correct, ax=ax).get_lines()[0].get_data()
                else:
                    correct_kde = None

                if len(incorrect) >= 2:
                    incorrect_kde = sns.kdeplot(incorrect, ax=ax).get_lines()[1].get_data()

                else:
                    incorrect_kde = None

                if visualize:
                    ax.clear()

                    # Plot histograms
                    sns.histplot(correct, kde=True, color='green', label='Correct', ax=ax)
                    sns.histplot(incorrect, kde=True, color='red', label='Incorrect', ax=ax)

                correct_interp = None
                incorrect_interp = None

                if correct_kde is not None:
                    correct_interp = np.interp(x_vals, correct_kde[0], correct_kde[1])

                if incorrect_kde is not None:
                    incorrect_interp = np.interp(x_vals, incorrect_kde[0], incorrect_kde[1])


                if correct_interp is not None and incorrect_interp is not None:
                    overlap_vals = np.minimum(correct_interp, incorrect_interp)
                    total_overlap_area = np.trapz(overlap_vals, x_vals)  # Compute area of overlap using np.trapz

                    thresholds[(method, cls)] = x_vals[np.argmax(overlap_vals)]  # Minimizes overlap

                    if visualize:
                        # Add threshold to the plot
                        threshold = thresholds[(method, cls)]
                        ax.axvline(x=threshold, color='black', linestyle='dashed', linewidth=2, label='Threshold')
                        ax.text(threshold, 0.5 * ax.get_ylim()[1], f'{threshold:.2f}', color='black', fontsize=10, ha='center', va='bottom', fontweight='bold')

                if visualize:
                    if col == 0:     # Set titles and labels
                        ax.set_ylabel(method, fontsize=12, fontweight='bold')
                    if row == 0:
                        ax.set_title(f'Class {cls}', fontsize=12, fontweight='bold')

                    ax.legend()
    else:
        for row, method in enumerate(scoring_methods):
            # doctor alpha and not already normalized
            if method == "doctor_alpha" and dataframe["doctor_alpha"].min() >= -(1 + 1e-6) and dataframe["doctor_alpha"].max() <= 0:
                dataframe['doctor_alpha'] = normalize_doctor_alpha(dataframe)

            if visualize:
                # Get the appropriate axis for this subplot
                ax = axes[row]
            else:
                ax = None

            if method == "mcmc_soft_scores" or method == "mcmc_entropy_scores":
                correct = dataframe[dataframe['mcmc_predictions'] == dataframe['Targets']][method]
                incorrect = dataframe[dataframe['mcmc_predictions'] != dataframe['Targets']][method]

            elif method == "Laplace_score":
                correct = dataframe[dataframe['Laplace_predictions'] == dataframe['Laplace_targets']][method]
                incorrect = dataframe[dataframe['Laplace_predictions'] != dataframe['Laplace_targets']][method]

            elif swag and method == "SWAG_score":
                correct = dataframe[dataframe['SWAG_targets'] == dataframe['SWAG_predictions']][method]
                incorrect = dataframe[dataframe['SWAG_targets'] != dataframe['SWAG_predictions']][method]


            else:
                # Separate the data into correct and incorrect classifications for the current scoring method
                correct = dataframe[dataframe['IsCorrect'] == True][method]
                incorrect = dataframe[dataframe['IsCorrect'] == False][method]


            # Find the optimal threshold using KDE without clearing histograms
            x_vals = np.linspace(min(dataframe[method]), max(dataframe[method]), 1000)

            if len(correct) >= 2:
                correct_kde = sns.kdeplot(correct, ax=ax).get_lines()[0].get_data()
            else:
                correct_kde = None

            if len(incorrect) >= 2:
                incorrect_kde = sns.kdeplot(incorrect, ax=ax).get_lines()[1].get_data()

            else:
                incorrect_kde = None

            if visualize:
                ax.clear()
                sns.histplot(correct, kde=True, color='green', label='Correct', ax=ax)
                sns.histplot(incorrect, kde=True, color='red', label='Incorrect', ax=ax)

            correct_interp = None
            incorrect_interp = None

            if correct_kde is not None:
                correct_interp = np.interp(x_vals, correct_kde[0], correct_kde[1])

            if incorrect_kde is not None:
                 incorrect_interp = np.interp(x_vals, incorrect_kde[0], incorrect_kde[1])

            if correct_interp is not None and incorrect_interp is not None:
                overlap_vals = np.minimum(correct_interp, incorrect_interp)
                total_overlap_area = np.trapz(overlap_vals, x_vals)  # Compute area of overlap using np.trapz
                thresholds[method] = x_vals[np.argmax(overlap_vals)]  # Minimizes overlap

                if visualize:
                    # Add threshold to the plot
                    threshold = thresholds[method]
                    ax.axvline(x=threshold, color='black', linestyle='dashed', linewidth=2, label='Threshold')
                    ax.text(threshold, 0.5 * ax.get_ylim()[1], f'{threshold:.2f}', color='black', fontsize=10, ha='center', va='bottom', fontweight='bold')

            if visualize:
                # Set titles and labels
                ax.set_ylabel(method, fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)

    if visualize:
        plt.tight_layout(rect=[0, 0, 1, 0.98])    # Ensures better layout of the plots
        plt.show()

    return thresholds

def normalize_doctor_alpha(dataframe):
    """
    Normalize doctor_alpha scores to [0, 1]
    :param dataframe: pandas dataframe containing doctor_alpha scores
    :return: numpy array of doctor_alpha scores normalized to [0, 1]
    """
    df_copy = dataframe.copy()
    scores = df_copy['doctor_alpha'].to_numpy()
    num_classes = len((df_copy['Targets'].unique()))
    min_doctor_alpha = 1 - num_classes
    max_doctor_alpha = 0
    normalized_scores =  (scores - min_doctor_alpha) / (max_doctor_alpha - min_doctor_alpha)
    return normalized_scores


