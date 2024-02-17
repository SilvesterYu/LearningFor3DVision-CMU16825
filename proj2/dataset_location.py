# specify the root location where u downloaded the dataset
root_location = None
use_full_dataset = False
dataset_name = (
    "r2n2_shapenet_dataset_full" if use_full_dataset else "r2n2_shapenet_dataset"
)

R2N2_PATH = f"{root_location}/{dataset_name}/r2n2"
SHAPENET_PATH = f"{root_location}/{dataset_name}/shapenet"

if use_full_dataset:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_3c.json"  # split file contains data entry for 3 classes
else:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_03001627.json"  # split file contains data entry for 03001627 class
