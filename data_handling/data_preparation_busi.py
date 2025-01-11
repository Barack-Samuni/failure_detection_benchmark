from pathlib import Path
import random
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

#import numpy as np
#import os
#import cv2


def delete_busi_mask_files(dataset_dir, mask_suffix="_mask.png"):
    """
    Deletes mask files with specific suffixes or containing '_mask' in their names
    from the 'benign', 'malignant', and 'normal' directories in the BUsI dataset.

    Args:
        dataset_dir (str or Path): Path to the root directory of the BUsI dataset.
        mask_suffix (str): Suffix used to identify mask files (e.g., "_mask.png").
    """
    dataset_path = Path(dataset_dir)

    # Directories to search for masks
    subdirs = ["benign", "malignant", "normal"]

    for subdir in subdirs:
        subdir_path = dataset_path / subdir
        if not subdir_path.is_dir():
            print(f"Warning: {subdir_path} does not exist or is not a directory.")
            continue

        # Delete mask files with the specific suffix
        for mask_file in subdir_path.rglob(f"*{mask_suffix}"):
            try:
                mask_file.unlink()
                print(f"Deleted (suffix match): {mask_file}")
            except Exception as e:
                print(f"Failed to delete {mask_file}: {e}")

        # Delete mask files containing '_mask' in their names
        for mask_file in subdir_path.rglob("*_mask*"):
            try:
                mask_file.unlink()
                print(f"Deleted (contains '_mask'): {mask_file}")
            except Exception as e:
                print(f"Failed to delete {mask_file}: {e}")




def count_files_in_busi_dirs(dataset_dir):
    """
    Counts the number of files in each directory of the BUsI dataset (benign, malignant, normal).

    Args:
        dataset_dir (str or Path): Path to the root directory of the BUsI dataset.
    """
    dataset_path = Path(dataset_dir)
    subdirs = ["benign", "malignant", "normal"]

    for subdir in subdirs:
        subdir_path = dataset_path / subdir
        if subdir_path.is_dir():
            file_count = sum(1 for _ in subdir_path.rglob("*") if _.is_file())
            print(f"Directory '{subdir}' contains {file_count} files.")
        else:
            print(f"Directory '{subdir}' does not exist.")


#######################################################
############## Augmentation functions #################
######################################################

# Augmentation Function
def augment_image(image, augmentations):
    """
    Applies a set of augmentations to an image.
    """
    for aug in augmentations:
        image = aug(image)
    return image

# Augmentation and Save Function with Visualization
def augment_and_save(dataset_dir, target_count, augmentations, visualize=False):
    """
    Augments images in the BUsI dataset to balance classes and optionally visualize results.
    """
    dataset_path = Path(dataset_dir)
    subdirs = ["benign", "malignant", "normal"]

    for subdir in subdirs:
        subdir_path = dataset_path / subdir
        if not subdir_path.is_dir():
            print(f"Warning: {subdir_path} does not exist. Skipping.")
            continue

        # Count existing files
        files = list(subdir_path.glob("*.png"))
        existing_count = len(files)

        if existing_count >= target_count:
            print(f"Directory '{subdir}' already contains {existing_count} files. No augmentation needed.")
            continue

        print(f"Augmenting '{subdir}' to reach {target_count} files...")

        # Visualization preparation
        if visualize:
            selected_files = random.sample(files, 3)  # Select 3 random files to visualize

        # Generate new files
        new_count = target_count - existing_count
        for i in range(new_count):
            # Select a random image
            source_file = random.choice(files)
            image = Image.open(source_file)

            # Apply augmentations
            augmented_image = augment_image(image, augmentations)

            # Save the augmented image with a unique name
            new_filename = f"{subdir} ({existing_count + i + 1}).png"
            augmented_image.save(subdir_path / new_filename)

        print(f"Directory '{subdir}' now contains {target_count} files.")

        # Visualization for this class
        if visualize:
            print(f"Visualizing augmentations for class '{subdir}'...")
            visualize_augmentations(subdir, selected_files, augmentations)

def visualize_augmentations(class_name, image_files, augmentations):
    """
    Visualizes the original image and three augmentations side by side.
    """
    fig, axes = plt.subplots(len(image_files), 4, figsize=(16, len(image_files) * 4))
    fig.suptitle(f"Augmentations for Class: {class_name}", fontsize=16)

    for idx, file in enumerate(image_files):
        original_image = Image.open(file)

        # Display original image
        axes[idx, 0].imshow(original_image, cmap="gray")
        axes[idx, 0].set_title("Original")
        axes[idx, 0].axis("off")

        # Display augmentations
        for aug_idx in range(1, 4):
            augmented_image = augment_image(original_image.copy(), augmentations[:aug_idx])
            axes[idx, aug_idx].imshow(augmented_image, cmap="gray")
            axes[idx, aug_idx].set_title(f"Aug {aug_idx}")
            axes[idx, aug_idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Define Augmentations
def random_horizontal_flip(image):
    if random.random() > 0.5:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def random_rotation(image):
    angle = random.randint(-10, 10)
    return image.rotate(angle)

def random_color_jitter(image):
    enhancer = ImageEnhance.Color(image)
    factor = random.uniform(0.8, 1.2)  # Slight color variation
    return enhancer.enhance(factor)



#########################################################
####### Main function for the BUSI data preparation #####
#########################################################

def process_busi_dataset(
        dataset_dir,
        mask_suffix="_mask.png",
        target_count=500,
        augmentations=None,
        visualize=True
):
    """
    Processes the BUSI dataset by removing unnecessary mask files,
    applying augmentations, and counting files before and after augmentation.

    Parameters:
        dataset_dir (str): Path to the dataset directory.
        mask_suffix (str): Suffix for mask files to delete. Default is "_mask.png".
        target_count (int): Number of augmented images to target per class. Default is 500.
        augmentations (list): List of augmentation functions to apply. Default is None.
        visualize (bool): Whether to visualize augmentations. Default is True.
    """
    # Step 1: Delete unnecessary mask files
    print("\nDeleting unnecessary mask files...")
    delete_busi_mask_files(dataset_dir, mask_suffix)

    # Step 2: Count files before augmentation
    print("\nNumber of files before augmentations:\n")
    count_files_in_busi_dirs(dataset_dir)

    # Step 3: Apply augmentations
    if augmentations is None:
        augmentations = [random_horizontal_flip, random_rotation, random_color_jitter]
    print("\nApplying augmentations...")
    augment_and_save(dataset_dir, target_count, augmentations, visualize=visualize)

    # Step 4: Count files after augmentation
    print("\nNumber of files after augmentations:\n")
    count_files_in_busi_dirs(dataset_dir)


"""
Example usage:

import data_preparation_busi as dp
dataset_dir = "Dataset_BUSI_with_GT"  # Replace with the actual path
augmentations_to_do = [dp.random_horizontal_flip, dp.random_rotation, dp.random_color_jitter]


dp.process_busi_dataset(
    dataset_dir = dataset_dir,
    mask_suffix = "_mask.png",
    target_count = 500,
    augmentations = augmentations_to_do,
    visualize = True
)
"""

if __name__ == "__main__":
    from default_paths import DATA_BUSI
    process_busi_dataset(
        dataset_dir = DATA_BUSI,
        mask_suffix = "_mask.png",
        target_count = 500,
        augmentations = [random_horizontal_flip, random_rotation, random_color_jitter],
        visualize = True
    )