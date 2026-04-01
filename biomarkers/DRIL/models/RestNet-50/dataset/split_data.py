import os
import random
import shutil

# ===== CONFIG =====
source_dir = r"C:\Users\hp Probook\Desktop\tryGit\DME-Biomarker-Detection\biomarkers\DRIL\models\RestNet-50\dataset\data_raw"          # contains DRIL / non DRIL
output_dir = r"C:\Users\hp Probook\Desktop\tryGit\DME-Biomarker-Detection\biomarkers\DRIL\models\RestNet-50\dataset\data" 

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

# ===== CLASS MAPPING =====
class_map = {
    "non_DRIL": "class0",
    "DRIL": "class1"
}

# ===== CREATE FOLDERS =====
for split in ["train", "val", "test"]:
    for cls in ["class0", "class1"]:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# ===== FUNCTION =====
def split_and_copy(files, src_folder, dst_base, class_name):
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split, split_files in splits.items():
        for f in split_files:
            src_path = os.path.join(src_folder, f)
            dst_path = os.path.join(dst_base, split, class_name, f)
            shutil.copy2(src_path, dst_path)

# ===== MAIN =====
for src_class, dst_class in class_map.items():
    src_folder = os.path.join(source_dir, src_class)

    files = [
        f for f in os.listdir(src_folder)
        if os.path.isfile(os.path.join(src_folder, f))
    ]

    print(f"{src_class}: {len(files)} images")

    split_and_copy(files, src_folder, output_dir, dst_class)

print("Done splitting dataset.")