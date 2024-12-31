from torch.utils.data import DataLoader, TensorDataset

from failure_detection.uncertainty_scores.softmax_scorer import (
    get_threshold,
    SoftmaxBasedScorer,
)
from laplace import Laplace
import torch
from torch.utils.data import DataLoader, Dataset


# Preprocess Dataset Targets
class TargetLongDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        X, y = self.original_dataset[index]
        # Ensure targets are torch.long
        y = torch.tensor(y, dtype=torch.long) if not isinstance(y, torch.Tensor) or y.dtype != torch.long else y
        return X, y

    def __len__(self):
        return len(self.original_dataset)

def get_laplace_scores(model, train_loader, test_loader, val_loader, n_classes):
    # Create new DataLoaders with processed datasets
    def create_new_loader(old_loader):
        return DataLoader(
            dataset=TargetLongDataset(old_loader.dataset),  # Wrap dataset
            batch_size=old_loader.batch_size,
            shuffle=old_loader.batch_sampler is None or getattr(old_loader.batch_sampler, "shuffle", False),
            num_workers=old_loader.num_workers,
            pin_memory=old_loader.pin_memory,
            drop_last=old_loader.drop_last,
        )

    # Replace original loaders with new ones
    train_loader = create_new_loader(train_loader)
    test_loader = create_new_loader(test_loader)
    val_loader = create_new_loader(val_loader)

    # Fit Laplace on training set
    la = Laplace(model, "classification")


    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik")

    # Get predictions on test dataloader
    test_probas, test_targets = [], []
    for img, target in test_loader:
        with torch.no_grad():
            if torch.cuda.is_available():
                img = img.cuda()
            result = la(img).cuda()  # .cuda()

            test_probas.append(result)  # laplace gives by probabilities
            test_targets.append(target)
    test_probas, test_targets = (
        torch.cat(test_probas).cuda(),
        torch.cat(test_targets).cuda(),
    )
    test_out = {"Laplace_targets": test_targets}

    # If binary, compute threshold on val set and get scores
    if n_classes <= 2:
        val_probas, val_targets = [], []
        for img, targets in val_loader:
            with torch.no_grad():
                if torch.cuda.is_available():
                    img = img.cuda()
                result = la(img).cuda()
                val_probas.append(result)  # laplace gives by probabilities
                val_targets.append(targets)
        val_probas, val_targets = (
            torch.cat(val_probas).cuda(),
            torch.cat(val_targets).cuda(),
        )
        threshold = get_threshold(val_probas, val_targets, target_fpr=0.2)
        test_out["Laplace_predictions"] = test_probas[:, 1] > threshold
        test_out["Laplace_threshold"] = threshold
        test_out["Laplace_score"] = SoftmaxBasedScorer(threshold).get_scores(
            test_probas
        )
        test_out["Laplace_probas"] = test_probas[:, 1]
    # Else, get scores
    else:
        test_out["Laplace_predictions"] = torch.argmax(test_probas, 1)
        test_out["Laplace_score"] = SoftmaxBasedScorer().get_scores(test_probas)
    # Memory management on GPU seems to run into some issue with the laplace object
    # these 2 lines havex fixed it.
    la.model = la.model.cuda()
    del la
    return test_out
