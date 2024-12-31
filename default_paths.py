from pathlib import Path

DATA_DIR_RSNA = Path(r"C:\Computer_Vision\Medical_Images_With_Rejection\failure_detection_benchmark\datasets\rsna-pneumonia-detection-challenge")
DATA_DIR_RSNA_PROCESSED_IMAGES = DATA_DIR_RSNA / "preprocess_224_224"

DATA_DIR_DIABETIC = Path(r"C:\Computer_Vision\Medical_Images_With_Rejection\failure_detection_benchmark\datasets\diabetic-retinopathy-detection")
DATA_DIR_DIABETIC_PROCESSED_IMAGES = DATA_DIR_DIABETIC / "preprocess_as_tensorflow"


DATA_MEDMNIST = Path(__file__).parent / "datasets"
DATA_BUSI = Path(r"C:\Computer_Vision\Medical_Images_With_Rejection\failure_detection_benchmark\datasets\Dataset_BUSI\Dataset_BUSI_with_GT")
