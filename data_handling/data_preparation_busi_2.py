
"""
######## Algorithm for augmentation ########

1.  create a train_data directory in the root_dir of the dataset. 
    If the directory already exists, delete the files inside this directory.

2.  get the list of files for the train from the variable  train_files, and copy them
    to train_data directory. 

3. get the labels for each file from y_train. count how many labels from each class.

4.  perform data augmentation by random_horizontal_flip,random_rotation,random_color_jitter.
    the data should be balanced in the end, so if there is a class with less samples, it should
    have more augmentations, meaning, there should be equal or almost equal amount of samples from each class.
    the variables y_train and train_files should keep track of the new files created, 
    so train_files will add the new file path created, and y_train will add the corresponding label to this new sample.
"""


from pathlib import Path
import numpy as np
import random
from PIL import Image
from typing import List, Optional
import shutil
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.io import imread


def create_and_augment_train_data(
    train_files: List[Path],
    y_train: np.ndarray,
    root_dir: Path,
    augmentation_transforms=None,
    plot_augmented: bool = False,
) -> tuple[List[Path], np.ndarray]:
    """
    Master function that creates the train_data directory, copies the train files,
    performs augmentations, and balances the dataset.

    Args:
        train_files (List[Path]): List of training data file paths.
        y_train (np.ndarray): Array of labels for the training data.
        root_dir (Path): Root directory of the dataset.
        augmentation_transforms: The transformations to apply for augmentations.
        plot_augmented (bool): Whether to plot augmented images.

    Returns:
        tuple: Updated lists of train files and labels.
    """
    print("Starting the data preparation process...")

    # Step 1: Create a train_data directory or clear it if already exists
    train_data_dir = create_train_data_directory(root_dir)

    # Step 2: Copy the original training files to the train_data directory
    copy_files_to_train_data(train_files, train_data_dir)

    # Step 3: Perform data augmentation and balance the dataset
    augmented_train_files, augmented_y_train = augment_and_balance_data(
        train_files, y_train, train_data_dir, augmentation_transforms, plot_augmented
    )

    # Final status prints
    print("\n=== Final Status ===")
    final_class_counts = count_labels_per_class(augmented_y_train)
    print(f"Final class counts: {final_class_counts}")
    print(f"Total training files: {len(augmented_train_files)}")
    print(f"Augmented train files: {augmented_train_files}")
    print(f"Augmented labels: {augmented_y_train.tolist()}")

    return augmented_train_files, augmented_y_train


def create_train_data_directory(root_dir: Optional[Path]) -> None:
    """
    Ensures the train_data directory under root_dir is ready for use:
    - If the directory exists, delete only the files inside it.
    - If the directory does not exist, create it.

    Args:
        root_dir (Optional[Path]): Root directory of the dataset.
    """
    if root_dir is None:
        raise ValueError("root_dir cannot be None. Please provide a valid directory path.")

    root_dir = Path(root_dir)  # Ensure it's a Path object
    train_data_dir = root_dir / "train_data"

    if train_data_dir.exists():
        print(f"Clearing files in the directory: {train_data_dir}")
        for item in train_data_dir.iterdir():
            if item.is_file():  # Only delete files
                item.unlink()
        print(f"Cleared all files in: {train_data_dir}")
    else:
        print(f"Creating directory: {train_data_dir}")
        train_data_dir.mkdir(parents=True, exist_ok=True)

    return train_data_dir

def copy_files_to_train_data(train_files: List[Path], train_data_dir: Path) -> None:
    """
    Copy the list of train files into the train_data directory.
    """
    print(f"Copying {len(train_files)} train files to {train_data_dir}...")
    for file in train_files:
        destination = train_data_dir / file.name
        shutil.copy(file, destination)


def count_labels_per_class(y_train: np.ndarray) -> dict:
    """
    Count the number of occurrences of each label in the y_train array.
    """
    return dict(zip(*np.unique(y_train, return_counts=True)))



def augment_and_balance_data(
    train_files: List[Path],
    y_train: np.ndarray,
    train_data_dir: Path,
    augmentation_transforms,
    plot_augmented: bool,
) -> tuple[List[Path], np.ndarray]:
    """
    Perform data augmentation to balance the training dataset.

    Args:
        train_files (List[Path]): List of original train files.
        y_train (np.ndarray): Array of labels for train files.
        train_data_dir (Path): Directory to store train data and augmentations.
        augmentation_transforms: Transformations to apply to augment data.
        plot_augmented (bool): Whether to plot augmentations (3 samples per class).

    Returns:
        tuple: Augmented train files and updated labels.
    """
    print("\n=== Data Augmentation ===")
    augmented_train_files = train_files.copy()
    augmented_y_train = y_train.copy()

    # Count the current class distribution
    class_counts = count_labels_per_class(y_train)
    print(f"Initial class counts: {class_counts}")

    # Determine the maximum class count for balancing
    max_class_count = max(class_counts.values())
    print(f"Maximum samples per class needed: {max_class_count}")

    # For plotting later: store one example for each class
    examples_for_plot = {class_label: [] for class_label in class_counts.keys()}

    # Perform augmentation for classes with fewer samples
    for class_label, count in class_counts.items():
        class_files = [train_files[i] for i in range(len(train_files)) if y_train[i] == class_label]
        num_augmentations_needed = max_class_count - count
        print(f"Class {class_label}: Needs {num_augmentations_needed} augmentations.")

        for i in range(num_augmentations_needed):
            original_file = random.choice(class_files)
            augmented_file_path = train_data_dir / f"{original_file.stem}_augmented_{i}{original_file.suffix}"

            # Open the image and apply augmentation
            image = Image.open(original_file)
            augmented_image = augmentation_transforms(image)

            # Save the augmented image
            augmented_image.save(augmented_file_path)

            # Update the lists with the new file and label
            augmented_train_files.append(augmented_file_path)
            augmented_y_train = np.append(augmented_y_train, class_label)

            # Add to examples for plotting (limit to 3 per class)
            if plot_augmented and len(examples_for_plot[class_label]) < 3:
                examples_for_plot[class_label].append((image, augmented_image))

    # Plot the examples if required
    if plot_augmented:
        plot_augmented_samples(examples_for_plot)

    return augmented_train_files, augmented_y_train







def plot_augmented_samples(examples_for_plot: dict) -> None:
    """
    Plot three samples of augmentations for each class.

    Args:
        examples_for_plot (dict): Dictionary mapping class labels to original-augmented image pairs.
    """
    print("\n=== Plotting Augmented Samples ===")
    for class_label, examples in examples_for_plot.items():
        if not examples:
            continue
        print(f"Class {class_label}: Plotting {len(examples)} samples.")
        fig, axes = plt.subplots(len(examples), 2, figsize=(8, 4 * len(examples)))
        fig.suptitle(f"Augmented Samples for Class {class_label}", fontsize=16)

        if len(examples) == 1:
            axes = [axes]  # Ensure axes are iterable for a single sample

        for i, (original, augmented) in enumerate(examples):
            axes[i][0].imshow(original, cmap="gray")
            axes[i][0].set_title("Original Image")
            axes[i][0].axis("off")

            axes[i][1].imshow(augmented, cmap="gray")
            axes[i][1].set_title("Augmented Image")
            axes[i][1].axis("off")

        plt.tight_layout()
        plt.show()

