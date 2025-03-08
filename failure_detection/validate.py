import numpy as np
from IPython.display import display, Markdown

from rejection.rejection import reject, rejection_confusion_matrix
from rejection.thresholding import find_thresholds
import pandas as pd
from rejection.confidence_score import find_optimal_weights, calculate_confidence_score
from pathlib import Path

def get_parameters_and_validate(train_seeds: np.ndarray, validation_seeds: np.ndarray) -> None:
    """
    This function fetches the thresholds from the train data of the train_seeds and validates it on the validation set
    from the validation seeds. The average AUC is calculated across the validation seeds.
    :param train_seeds: numpy array of train seeds to get parameters from
    :param validation_seeds: numpy array of validation seeds to validate on
    """
    # ---------- first load the train seed and find thresholds and optimal weights from it
    auc_seeds = {"Seed": np.array([]), "Average_AUC": np.array([])}
    root_dir = Path(__file__).parent.parent.resolve()

    for train_seed in train_seeds:
        auc_seeds["Seed"] = np.append(auc_seeds["Seed"], train_seed)
        auc_vec = np.array([])

        # -----------Thresholding----------------
        display(Markdown(f"## **Train seed: {train_seed}**\n***"))
        display(Markdown("### **Thresholding**\n***"))
        train_scores_df = pd.read_csv(Path(f"{root_dir}/outputs_train/RSNAPneumonia/resnet50/dropout_all_layers_autolr_paper"
                                      f"/seed_{train_seed}/failure_detection/scores_df.csv"))
        thresholds = find_thresholds(train_scores_df, visualize=True, separate_classes=False, confidnet=True, swag=True)
        thresholds_without_classes_separation_dict = {"Seed": len(thresholds.keys()) * [train_seed],
                                                      "Scoring_Method": [method for method in
                                                                         thresholds.keys()],
                                                      "Threshold": thresholds.values()}
        thresholds_without_class_separation_df = pd.DataFrame(thresholds_without_classes_separation_dict)

        # Drop the rows where Scoring_Method is 'TrustScore' or 'mcmc_entropy_scores'
        filtered_thresholds_without_class_separation_df = thresholds_without_class_separation_df[
            ~thresholds_without_class_separation_df['Scoring_Method'].isin(['TrustScore', 'mcmc_entropy_scores'])
        ]
        filtered_thresholds_without_class_separation_df = filtered_thresholds_without_class_separation_df.reset_index(
            drop=True)

        display(filtered_thresholds_without_class_separation_df)

        # -------------Finding Optimal weights-----------------
        display(Markdown('### **Finding Optimal weights**\n***'))
        weights_df = find_optimal_weights(dataframe=train_scores_df,
                                          thresholds=filtered_thresholds_without_class_separation_df)
        display(weights_df)

        # ------------Validation process-----------------------
        display(Markdown('### **Validation process**\n***'))

        for validation_seed in validation_seeds:
            display(Markdown(f"#### **Validation seed: {validation_seed}**\n***"))
            validation_scores_df = pd.read_csv(Path(f"{root_dir}/outputs/RSNAPneumonia/resnet50"
                                                    f"/dropout_all_layers_autolr_paper/seed_{validation_seed}"
                                                    f"/failure_detection/scores_df.csv"))

            # -----------------Calculate confidence score---------------------------
            display(Markdown("##### **Calculate confidence score**\n***"))
            confidence_scores_df = calculate_confidence_score(dataframe=validation_scores_df, weights=weights_df)
            display(confidence_scores_df)

            # -------------------------Reject-----------------------------------------------
            display(Markdown('##### **Rejection**\n***'))
            confidence = np.linspace(0, 100, 101)
            graph_df, lowest_rejections, auc = reject(dataframe=confidence_scores_df, confidence_level=confidence,
                                                 visualize=True, output_points=True, get_lowest_rejection=True)
            display(Markdown('##### **Confusion Matrix**\n***'))
            rejection_confusion_matrix(dataframe=confidence_scores_df,
                                       confidence_level=lowest_rejections["Confidence Level"].iloc[1])
            auc_vec = np.append(auc_vec, auc)
        auc_seeds["Average_AUC"] = np.append(auc_seeds["Average_AUC"], np.mean(auc_vec))

    display(Markdown("## **Average AUC summary**\n***"))
    auc_df = pd.DataFrame(auc_seeds)
    display(auc_df)
    display(Markdown(f"Best seed is: {auc_df.loc[auc_df['Average_AUC'].idxmax(),'Seed']} with AUC: {auc_df['Average_AUC'].max():.2f}"))








