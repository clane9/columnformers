#!/usr/bin/env python3

import os
import json
import shutil
import argparse
import scipy.io as sio
from pathlib import Path
from collections import defaultdict


def process_val_labels(devkit_path, val_path):
    """
    Read validation ground truth and organize validation images into class folders

    Args:
        devkit_path (Path): Path to ILSVRC2012_devkit_t12 directory
        val_path (Path): Path to directory containing validation images
    """
    print(f"Processing validation images in: {val_path}")

    # Read validation labels
    val_labels_path = devkit_path / "data" / "ILSVRC2012_validation_ground_truth.txt"
    with open(val_labels_path, "r") as f:
        val_labels = [int(line.strip()) for line in f.readlines()]

    # Read meta.mat
    meta = sio.loadmat(str(devkit_path / "data" / "meta.mat"))
    synsets = meta["synsets"]

    # Create ILSVRC2012_ID -> WNID (synset) mapping
    synset_mapping = {int(s[0][0][0][0]): str(s[0][1][0]) for s in synsets}

    # Create mapping of filename to class
    val_filename_to_class = {}
    for i, label in enumerate(val_labels, 1):
        filename = f"ILSVRC2012_val_{i:08d}.JPEG"
        val_filename_to_class[filename] = synset_mapping[label]

    # Create temporary directory for moving files
    temp_dir = val_path / "temp_organization"
    os.makedirs(temp_dir, exist_ok=True)

    # First, create all class directories in temp
    for class_id in set(val_filename_to_class.values()):
        os.makedirs(temp_dir / class_id, exist_ok=True)

    # Move files to their class directories in temp
    print("Moving files to their class directories...")
    moved_count = 0
    total_files = len(val_filename_to_class)

    for filename, class_id in val_filename_to_class.items():
        src_path = val_path / filename
        dst_dir = temp_dir / class_id
        dst_path = dst_dir / filename

        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            moved_count += 1
            if moved_count % 1000 == 0:
                print(f"Moved {moved_count}/{total_files} files...")

    # Remove any remaining files in val_path (except temp_organization)
    for item in val_path.iterdir():
        if item.name != "temp_organization":
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Move everything from temp back to val_path
    for class_dir in temp_dir.iterdir():
        if class_dir.is_dir():
            shutil.move(str(class_dir), str(val_path / class_dir.name))

    # Remove temp directory
    shutil.rmtree(temp_dir)

    print("\nOrganization complete!")
    print(
        f"Moved {moved_count} files into {len(set(val_filename_to_class.values()))} class directories"
    )


def create_class_info(devkit_path, val_path):
    """
    Create a JSON file mapping synset IDs to human-readable labels and metadata

    Args:
        devkit_path (Path): Path to ILSVRC2012_devkit_t12 directory
        val_path (Path): Path to directory containing validation images
    """
    class_info = defaultdict(dict)

    # Read meta.mat
    meta = sio.loadmat(str(devkit_path / "data" / "meta.mat"))
    synsets = meta["synsets"]

    # Extract information for each synset
    for s in synsets:
        synset_id = str(s[0][1][0])  # WNID
        class_id = int(s[0][0][0][0])  # ILSVRC2012_ID
        words = str(s[0][2][0]).split(", ")
        gloss = str(s[0][3][0])

        class_info[synset_id].update(
            {"class_id": class_id, "words": words, "gloss": gloss}
        )

    # Count validation images per class
    if val_path.exists():
        for class_dir in val_path.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.JPEG")))
                class_info[class_dir.name]["val_images"] = count

    # Save to JSON file
    output_file = val_path.parent / "imagenet_class_info.json"
    with open(output_file, "w") as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)

    # Print example entries
    print("\nExample class mappings:")
    print("-" * 80)
    for synset_id in list(class_info.keys())[:5]:
        print(f"Synset ID: {synset_id}")
        print(f"Class ID: {class_info[synset_id]['class_id']}")
        print(f"Labels: {', '.join(class_info[synset_id]['words'])}")
        print(f"Description: {class_info[synset_id]['gloss']}")
        print(f"Validation images: {class_info[synset_id].get('val_images', 0)}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Process ImageNet labels and create class mappings"
    )
    parser.add_argument(
        "devkit_path", type=str, help="Path to ILSVRC2012_devkit_t12 directory"
    )
    parser.add_argument(
        "val_path", type=str, help="Path to directory containing validation images"
    )
    args = parser.parse_args()

    devkit_path = Path(args.devkit_path)
    val_path = Path(args.val_path)

    # Verify paths and files
    if not devkit_path.exists():
        raise FileNotFoundError(f"Devkit path does not exist: {devkit_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation images path does not exist: {val_path}")
    if not (devkit_path / "data" / "meta.mat").exists():
        raise FileNotFoundError(f"Could not find meta.mat in {devkit_path}/data")
    if not (devkit_path / "data" / "ILSVRC2012_validation_ground_truth.txt").exists():
        raise FileNotFoundError(
            f"Could not find validation ground truth file in {devkit_path}/data"
        )

    # Check if validation directory contains images
    val_images = list(val_path.glob("ILSVRC2012_val_*.JPEG"))
    if not val_images:
        raise FileNotFoundError(f"No validation images found in {val_path}")

    print("Organizing validation images into class folders...")
    process_val_labels(devkit_path, val_path)

    print("\nCreating class information JSON file...")
    create_class_info(devkit_path, val_path)

    print(
        f"\nDone! Check {val_path.parent}/imagenet_class_info.json for complete class mappings."
    )


if __name__ == "__main__":
    main()

# python3 process_imagenet_labels.py /weka/proj-medarc/shared/imagenet/ILSVRC2012_devkit_t12 /weka/proj-medarc/shared/imagenet/validation
