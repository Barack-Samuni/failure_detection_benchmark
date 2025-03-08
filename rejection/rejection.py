from matplotlib import pyplot as plt
from numpy import ndarray
from typing import Union, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from sklearn.metrics import confusion_matrix

def reject(dataframe,confidence_level: Union[float , ndarray] = 0.0, visualize: bool=False,
           output_points:bool=False, get_lowest_rejection: bool = False) \
        -> Union[None, Tuple[DataFrame,float],Tuple[DataFrame,DataFrame,float]]:
    """
    This function rejects every sample below the specified confidence level.
    :param dataframe: scores dataframe with confidence score
    :param confidence_level: confidence level to reject samples below it
    :param visualize: whether to plot the rejection rate Vs. accuracy
    :param output_points: whether to output the points of rejection rate and accuracy from the graph
    :param get_lowest_rejection: whether to output the lowest rejection rates and their maximal accuracies
    :return: max_accuracy, rejection_rate at that accuracy, area below graph
    """
    accuracy = np.array([])
    rejection_rate = np.array([])

    for confidence in confidence_level:
        # first calculate the rejection rate
        if confidence == 0:         # reject everything
            rejection = 100
            acc = 0

        elif confidence == 100:     # accept everything
            rejection = 0
            correct_condition = ((((dataframe["IsCorrect"] == True) |
                                 (dataframe["SWAG_predictions"] == dataframe["SWAG_targets"])) |
                                 (dataframe["Laplace_predictions"] == dataframe["Laplace_targets"])) |
                                 (dataframe["mcmc_predictions"] == dataframe["Targets"]))

            correct_samples = dataframe[correct_condition]
            acc = (correct_samples.shape[0] / dataframe.shape[0]) * 100

        else:
            rejection_threshold = 100 - confidence
            rejected_samples = dataframe[dataframe["Confidence_score"] < rejection_threshold]
            rejection = (rejected_samples.shape[0] / dataframe.shape[0]) * 100
            non_rejected_samples = dataframe[dataframe["Confidence_score"] >= rejection_threshold]
            correct_condition = ((((non_rejected_samples["IsCorrect"] == True) |
                                 (non_rejected_samples["SWAG_predictions"] == non_rejected_samples["SWAG_targets"])) |
                                 (non_rejected_samples["Laplace_predictions"] == non_rejected_samples["Laplace_targets"])) |
                                 (non_rejected_samples["mcmc_predictions"] == non_rejected_samples["Targets"]))

            correct_samples = non_rejected_samples[correct_condition]
            if non_rejected_samples.shape[0] == 0:      # rejected everything
                acc = 0
            else:
                acc = (correct_samples.shape[0] / non_rejected_samples.shape[0]) * 100

        rejection_rate = np.append(rejection_rate, rejection)
        accuracy = np.append(accuracy, acc)

    area_below_graph = np.abs(np.trapz(accuracy / 100, x=rejection_rate / 100))

    if visualize:
        plt.title(f"Accuracy Vs. Rejection Rate, AUC = {area_below_graph:.2f}")
        plt.plot(rejection_rate, accuracy, marker='o', color='b', linestyle='-')
        plt.xlabel("Rejection Rate[%]")
        plt.ylabel("Accuracy[%]")
        plt.show()

    if output_points:
        output_df = pd.DataFrame({"Confidence Level":confidence_level,"Rejection_Rate[%]": rejection_rate,
                                  "Accuracy[%]": accuracy})

        if get_lowest_rejection:
            lowest_rejection = get_lowest_rejection_rates(output_df, visualize=visualize)
            return output_df, lowest_rejection, area_below_graph

        return output_df, area_below_graph

def get_lowest_rejection_rates(output_points_df: DataFrame, n:int = 10, visualize: bool = False) -> Union[None, DataFrame]:
    """
    :param output_points_df: dataframe with the output points of rejection rates and accuracies
    :param visualize: whether to plot the lowest rejection rates and their maximal accuracies
    :param n: number of lowest rejection rates to plot and output
    :return: None, but plots the lowest rejection rates and their maximal accuracies if visualize is True.
             or else it returns the lowest rejection rates and their maximal accuracies.
    """
    unique_rejections = output_points_df.sort_values(by="Accuracy[%]", ascending=False).drop_duplicates(
        subset="Rejection_Rate[%]", keep="first")  # keep maximal accuracy per rejection rate
    unique_rejection_n_minimal = unique_rejections.nsmallest(n, "Rejection_Rate[%]")

    if visualize:
        plt.figure(figsize=(10, 6))
        sns.heatmap(unique_rejection_n_minimal, annot=True, cmap="inferno", fmt=".2f", cbar=True,
                    annot_kws={"size": 10})
        plt.title("Lowest Rejection Rates and their maximal Accuracy")
        plt.show()

    return unique_rejection_n_minimal

def rejection_confusion_matrix(dataframe: DataFrame, confidence_level: float) -> None:
    """
    This function plots the confusion matrix for the results in a form of Rejected Non-Rejected VS. Right and wrong
    descision.
    :param dataframe: dataframe with the results and confidence scores
    :param confidence_level: confidence level to reject samples below it
    """
    rejection_threshold = 100 - confidence_level
    dataframe["Rejected"] = np.where(dataframe["Confidence_score"] < rejection_threshold, "Rejected", "Not Rejected")

    correct_condition = ((((dataframe["IsCorrect"] == True) |
                           (dataframe["SWAG_predictions"] == dataframe["SWAG_targets"])) |
                          (dataframe["Laplace_predictions"] == dataframe["Laplace_targets"])) |
                         (dataframe["mcmc_predictions"] == dataframe["Targets"]))

    rejected_condition = dataframe["Rejected"] == "Rejected"
    dataframe["Right_Decision"] = np.where((correct_condition ^ rejected_condition), "Right_Decision", "Wrong_Decision")
    cm = confusion_matrix(dataframe["Rejected"].map({"Rejected": True, "Not Rejected": False}),
                          dataframe["Right_Decision"].map({"Right_Decision": True, "Wrong_Decision": False}))
    cm = (cm / cm.sum()) * 100
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="inferno", cbar=True, annot_kws={"size": 10},
                yticklabels=["Not Rejected", "Rejected"], xticklabels=["Wrong Decision", "Right Decision"])
    plt.title(f"Rejected Non-Rejected VS. Right and Wrong Decision, Confidence Level = {confidence_level}%")
    plt.show()


