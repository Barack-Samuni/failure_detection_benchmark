
import confidence_score as cs
import scores_constants
# Example usage
if __name__ == "__main__":

    train_root = r"C:\Users\97254\Desktop\niv\AFEKA\M.sc Machine learning\AI 2th year\Computer vision\failure_detection_benchmark-main\github_rep_failure_detection\train_seeds"
    val_root = r"C:\Users\97254\Desktop\niv\AFEKA\M.sc Machine learning\AI 2th year\Computer vision\failure_detection_benchmark-main\github_rep_failure_detection\val_seeds"

    best_thresholds, best_auc = cs.find_best_thresholds(scores_constants.equal_weights,train_root, val_root)
    print(best_auc)