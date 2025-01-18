import numpy as np
from sklearn.model_selection import train_test_split
from default_paths import DATA_BUSI
from torchvision import transforms

from data_handling import data_preparation_busi_2 as dp2
########## COPY OF data_handling/busi.py section ########
root_dir = DATA_BUSI

normal_filenames = list((root_dir / "normal").glob("*).png"))
benign_filenames = list((root_dir / "benign").glob("*).png"))
malignant_filenames = list((root_dir / "malignant").glob("*).png"))

train_val_normal, test_normal = train_test_split(
    normal_filenames, test_size=0.2, shuffle=True
)
train_normal, val_normal = train_test_split(
    train_val_normal, test_size=0.125, shuffle=True
)
y_train_normal, y_val_normal, y_test_normal = (
    np.repeat(0, len(train_normal)),
    np.repeat(0, len(val_normal)),
    np.repeat(0, len(test_normal)),
)

train_val_benign, test_benign = train_test_split(
    benign_filenames, test_size=0.2, shuffle=True
)
train_benign, val_benign = train_test_split(
    train_val_benign, test_size=0.125, shuffle=True
)
y_train_benign, y_val_benign, y_test_benign = (
    np.repeat(1, len(train_benign)),
    np.repeat(1, len(val_benign)),
    np.repeat(1, len(test_benign)),
)

train_val_malignant, test_malignant = train_test_split(
    malignant_filenames, test_size=0.2, shuffle=True
)
train_malignant, val_malignant = train_test_split(
    train_val_malignant, test_size=0.125, shuffle=True
)
y_train_malignant, y_val_malignant, y_test_malignant = (
    np.repeat(2, len(train_malignant)),
    np.repeat(2, len(val_malignant)),
    np.repeat(2, len(test_malignant)),
)

train_files_temp = train_normal + train_benign + train_malignant
y_train_temp = np.concatenate((y_train_normal, y_train_benign, y_train_malignant))
###################################################################


# Example of augmentation transforms
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
])

# Call the master function
train_files, y_train = dp2.create_and_augment_train_data(
    train_files=train_files_temp,
    y_train=y_train_temp,
    root_dir=root_dir,
    augmentation_transforms=augmentation_transforms,
    plot_augmented=True  # Set to False if you don't want to plot augmentations
)
