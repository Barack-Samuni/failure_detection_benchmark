
"""
######## Algorithm for augmentation ########

1.  create a train_data directory in the root_dir of the dataset.
    In this directory create 3 directories: benign, malignant, normal.
    If the directories already exists, delete the files inside these directories.

2.  get the list of files for  train from the variable  train_files_temp, and copy them
    to the corresponding directory in train_data directory according to y_train_temp,
    which holds an ndarray with the corresponding class corresponding to train_files_temp,
    where:0 -  normal, 1 - benign, 2- malignant.

3. get the labels for each file from y_train_temp and count how many labels there are from each class.

4.  perform data augmentation by random_horizontal_flip,random_rotation,random_color_jitter.
    the data should be balanced in the end, so if there is a class with less samples, it should
    have more augmentations, meaning, there should be equal or almost equal amount of samples from each class.
    make option for visualisation of 3 examples of the augmentations created.

5. Create lists of file names from each class directory in the directory train_data, and create a corresponding ndarrays
    which holds the class number corresponding to the list.

6. concatenate the lists of files into one big list of files and  ndarrays to corresponding
    one big ndarray of class numbers.
    return the two variables.
    """


from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from torchvision import transforms
import random
import matplotlib.pyplot as plt

def setup_train_data_directory(root_dir, class_dirs):
    train_data_dir = Path(root_dir) / 'train_data'
    if train_data_dir.exists():
        shutil.rmtree(train_data_dir)
    train_data_dir.mkdir(parents=True)
    for class_dir in class_dirs:
        (train_data_dir / class_dir).mkdir()
    print(f"Created train_data directory with subdirectories: {class_dirs}")
    return train_data_dir

def copy_files_to_class_directories(train_files_temp, y_train_temp, train_data_dir, class_dirs):
    for file, label in zip(train_files_temp, y_train_temp):
        class_dir = class_dirs[label]
        shutil.copy(file, train_data_dir / class_dir)
    print("Copied training files to their respective class directories.")

def count_labels(y_train_temp, class_dirs):
    label_counts = {class_dir: 0 for class_dir in class_dirs}
    for label in y_train_temp:
        label_counts[class_dirs[label]] += 1
    print("Initial class counts:")
    for class_name, count in label_counts.items():
        print(f"  {class_name}: {count} images")
    return label_counts

def perform_data_augmentation(train_data_dir, class_dirs, max_count, visualize=False):
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
    ])

    for class_dir in class_dirs:
        class_path = train_data_dir / class_dir
        files = list(class_path.iterdir())
        current_count = len(files)
        augment_count = max_count - current_count
        print(f"Augmenting {augment_count} images for class '{class_dir}'...")

        if visualize:
            examples_to_plot = random.sample(files, min(3, len(files)))
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for _ in range(augment_count):
            file_to_augment = random.choice(files)
            img = Image.open(file_to_augment)
            augmented_img = augmentation_transforms(img)

            augmented_file_name = f"aug_{random.randint(0, 1e6)}_{file_to_augment.name}"
            augmented_img.save(class_path / augmented_file_name)

        if visualize:
            for i, example_file in enumerate(examples_to_plot):
                img = Image.open(example_file)
                augmented_img = augmentation_transforms(img)
                axes[i].imshow(augmented_img)
                axes[i].axis('off')
                axes[i].set_title(f"Augmented Example {i + 1}")
            plt.show()

    print("Data augmentation completed.")

def create_file_and_label_lists(train_data_dir, class_dirs):
    file_paths = []
    labels = []

    for label, class_dir in enumerate(class_dirs):
        class_path = train_data_dir / class_dir
        files = list(class_path.iterdir())

        file_paths.extend([str(file) for file in files])
        labels.extend([label] * len(files))

        print(f"Final count for class '{class_dir}': {len(files)} images")

    return file_paths, np.array(labels)


### MASTER FUNCTION FOR THE AUGMENTATION AND BALANCE ###
def augment_and_balance_dataset(root_dir, train_files_temp, y_train_temp, visualize=False):
    class_dirs = ['normal', 'benign', 'malignant']

    # Step 1: Setup train_data directory
    train_data_dir = setup_train_data_directory(root_dir, class_dirs)

    # Step 2: Copy files to corresponding directories
    copy_files_to_class_directories(train_files_temp, y_train_temp, train_data_dir, class_dirs)

    # Step 3: Count labels
    label_counts = count_labels(y_train_temp, class_dirs)
    max_count = max(label_counts.values())
    print(f"Target number of images per class after augmentation: {max_count}")

    # Step 4: Data augmentation
    perform_data_augmentation(train_data_dir, class_dirs, max_count, visualize)

    # Step 5 & 6: Create file and label lists
    file_paths, labels = create_file_and_label_lists(train_data_dir, class_dirs)

    print("Augmentation and balancing process completed.")
    return file_paths, labels
