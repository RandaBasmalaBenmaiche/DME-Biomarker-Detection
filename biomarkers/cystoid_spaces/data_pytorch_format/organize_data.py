from pathlib import Path
import shutil

root = Path(r"C:\Users\hp Probook\Desktop\tryGit\DME-Biomarker-Detection\biomarkers\cystoid_spaces\data_raw\kaggle\images-training-labels")
out_images = Path("all_images")
out_masks = Path("all_masks")

out_images.mkdir(exist_ok=True)
out_masks.mkdir(exist_ok=True)

for path in root.rglob("*"):
    if path.is_file():
        parent_name = path.parent.name.lower()

        if parent_name == "images":
            shutil.copy(path, out_images / path.name)

        elif parent_name == "masks":
            shutil.copy(path, out_masks / path.name)