"""
Dataset splitter — handles ALL observed naming conventions:

  Convention A (original):
    Image : {digits}{anything}         e.g. 6DME_as_F
    Mask  : {digits}MASK_{anything}    e.g. 6MASK_DME_as_F

  Convention B (suffix-mask):
    Image : dme_a10  /  DME_S_001      (any name NOT ending in 'mask')
    Mask  : dme_a10mask / DME_S_001mask (same stem + 'mask' suffix)

  Convention C (prefix-dash-mask):
    Image : DME84  /  DME85  ...
    Mask  : MASK-DME84  /  MASK-DME85  ... (case-insensitive 'MASK-' prefix)

Unmatched files are reported but never silently dropped.
Multiple masks per image are fully supported.
"""

import os
import re
import shutil
import random
import json
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────
# CONFIGURATION  ← edit these before running
# ──────────────────────────────────────────────
IMAGES_DIR  = r"C:\Users\hp Probook\Desktop\tryGit\DME-Biomarker-Detection\biomarkers\cystoid_spaces\data_pytorch_format\all_images"        # folder containing raw images
MASKS_DIR   = r"C:\Users\hp Probook\Desktop\tryGit\DME-Biomarker-Detection\biomarkers\cystoid_spaces\data_pytorch_format\all_masks"         # folder containing masks
OUTPUT_DIR  = r"C:\Users\hp Probook\Desktop\tryGit\DME-Biomarker-Detection\biomarkers\cystoid_spaces\data_pytorch_format\split_output"  # where train/ and val/ will be created
TRAIN_RATIO = 0.8             # 0.8 = 80 % train, 20 % val
RANDOM_SEED = 42
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
COPY_FILES  = True            # True = copy files; False = only write split.json
# ──────────────────────────────────────────────


# ── helpers ───────────────────────────────────────────────────────────────────

def normalise(stem: str) -> str:
    """Lower-case stem for case-insensitive comparisons."""
    return stem.lower()


def classify_mask(stem: str):
    """
    Given a mask file stem, return (convention, image_stem_lower) or None if
    the file doesn't look like a mask at all.

    Convention tags:
      'A'  – digits + MASK_ prefix (underscore)  e.g. 6MASK_DME_as_F  -> '6dme_as_f'
      'B'  – stem ending in 'mask'               e.g. dme_a10mask      -> 'dme_a10'
      'C'  – MASK- prefix (dash)                 e.g. MASK-DME84       -> 'dme84'
    """
    sl = normalise(stem)

    # Convention A: digits then MASK_ (underscore)
    m = re.match(r'^(\d+)(mask_)(.+)$', sl)
    if m:
        img_stem = m.group(1) + m.group(3)   # e.g. '6dme_as_f'
        return ('A', img_stem)

    # Convention C: MASK- prefix with dash (check before B to avoid 'mask-...' matching endswith)
    if sl.startswith('mask-'):
        img_stem = sl[5:]                     # strip 'mask-'
        return ('C', img_stem)

    # Convention B: stem ending with 'mask'
    if sl.endswith('mask'):
        img_stem = sl[:-4]                    # strip trailing 'mask'
        return ('B', img_stem)

    return None   # not a mask by our conventions


def gather_pairs(images_dir: Path, masks_dir: Path) -> dict:
    """
    Returns {image_filename: {"image": Path, "masks": [Path, ...]}}
    Only images with at least one matched mask are included.
    """

    # ── index images: normalised_stem -> Path ─────────────────────────────────
    images = {}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        key = normalise(p.stem)
        if key in images:
            print(f"  [WARN] Duplicate image stem (case-insensitive): {p.name} "
                  f"vs {images[key].name} — keeping first")
        else:
            images[key] = p

    # ── index masks: normalised_image_stem -> [mask Path] ─────────────────────
    mask_index = defaultdict(list)
    unclassified_masks = []

    for p in sorted(masks_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        result = classify_mask(p.stem)
        if result is None:
            unclassified_masks.append(p.name)
            continue
        _, img_stem_lower = result
        mask_index[img_stem_lower].append(p)

    if unclassified_masks:
        print(f"\n  [WARN] {len(unclassified_masks)} mask(s) don't match any known "
              f"naming convention (will be ignored):")
        for n in unclassified_masks:
            print(f"         {n}")

    # ── pair up ───────────────────────────────────────────────────────────────
    pairs = {}
    unmatched_images = []

    for img_key, img_path in images.items():
        if img_key in mask_index:
            pairs[img_path.name] = {
                "image": img_path,
                "masks": sorted(mask_index[img_key]),
            }
        else:
            unmatched_images.append(img_path.name)

    if unmatched_images:
        print(f"\n  [WARN] {len(unmatched_images)} image(s) have no matching mask "
              f"and will be excluded:")
        for n in unmatched_images[:20]:
            print(f"         {n}")
        if len(unmatched_images) > 20:
            print(f"         ... and {len(unmatched_images) - 20} more")

    # ── orphan masks (masks whose image wasn't found) ─────────────────────────
    matched_img_keys = {normalise(Path(v["image"]).stem) for v in pairs.values()}
    orphan_mask_keys = set(mask_index.keys()) - matched_img_keys
    if orphan_mask_keys:
        orphan_files = [mp.name for k in sorted(orphan_mask_keys)
                        for mp in mask_index[k]]
        print(f"\n  [WARN] {len(orphan_files)} mask file(s) have no matching image:")
        for n in orphan_files[:20]:
            print(f"         {n}")
        if len(orphan_files) > 20:
            print(f"         ... and {len(orphan_files) - 20} more")

    return pairs


# ── split & output ────────────────────────────────────────────────────────────

def split_pairs(pairs, train_ratio, seed):
    keys = sorted(pairs.keys())
    random.seed(seed)
    random.shuffle(keys)
    n_train = max(1, round(len(keys) * train_ratio))
    return keys[:n_train], keys[n_train:]


def copy_split(pairs, train_keys, val_keys, output_dir: Path):
    for split_name, keys in [("train", train_keys), ("val", val_keys)]:
        img_out  = output_dir / split_name / "images"
        mask_out = output_dir / split_name / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        for img_name in keys:
            entry = pairs[img_name]
            shutil.copy2(entry["image"], img_out / entry["image"].name)
            for mp in entry["masks"]:
                shutil.copy2(mp, mask_out / mp.name)
    print(f"\n  Files copied to: {output_dir}/")


def save_json(pairs, train_keys, val_keys, output_dir: Path):
    def serialise(keys):
        return [{"image": pairs[k]["image"].name,
                 "masks": [m.name for m in pairs[k]["masks"]]}
                for k in keys]

    out = output_dir / "split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"train": serialise(train_keys), "val": serialise(val_keys)},
                  f, indent=2)
    print(f"  Split manifest saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    images_dir = Path(IMAGES_DIR)
    masks_dir  = Path(MASKS_DIR)
    output_dir = Path(OUTPUT_DIR)

    for d, label in [(images_dir, "IMAGES_DIR"), (masks_dir, "MASKS_DIR")]:
        if not d.exists():
            raise FileNotFoundError(f"{label} not found: {d.resolve()}")

    print("── Gathering image-mask pairs ──────────────────────────────────────")
    pairs = gather_pairs(images_dir, masks_dir)

    if not pairs:
        print("No valid image-mask pairs found. Check your folder paths.")
        return

    total_masks = sum(len(v["masks"]) for v in pairs.values())
    print(f"\n  Paired images : {len(pairs)}")
    print(f"  Total masks   : {total_masks}")

    print("\n── Splitting ────────────────────────────────────────────────────────")
    train_keys, val_keys = split_pairs(pairs, TRAIN_RATIO, RANDOM_SEED)
    print(f"  Train : {len(train_keys)} images")
    print(f"  Val   : {len(val_keys)} images")

    save_json(pairs, train_keys, val_keys, output_dir)

    if COPY_FILES:
        print("\n── Copying files ────────────────────────────────────────────────────")
        copy_split(pairs, train_keys, val_keys, output_dir)

    print("\n── Sample train entries ─────────────────────────────────────────────")
    for k in train_keys[:3]:
        e = pairs[k]
        print(f"  image : {e['image'].name}")
        for m in e["masks"]:
            print(f"  mask  : {m.name}")
        print()

    print("── Done ✓ ───────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()