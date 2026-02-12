import os
import random
import shutil

import numpy as np
import pandas as pd


"""
Builds a 2-class dataset (healthy vs neurological_risk) from the PAPILA data.

Resulting structure (relative to this script's directory):

dataset_papila/
  train/
    healthy/
    neurological_risk/
  val/
    healthy/
    neurological_risk/

We treat diagnosis tag == 0 as "healthy" and any other tag as "neurological_risk".
"""


RANDOM_SEED = 42
TRAIN_SPLIT = 0.8

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PAPILA_ROOT = os.path.join(
    PROJECT_ROOT,
    "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f",
)
FUNDUS_DIR = os.path.join(PAPILA_ROOT, "FundusImages")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_papila")


def _fix_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate PAPILA helper to clean clinical Excel data."""
    df_new = df.drop(["ID"], axis=0)
    df_new.columns = df_new.iloc[0, :]
    df_new.drop([np.nan], axis=0, inplace=True)
    df_new.columns.name = "ID"
    return df_new


def get_diagnosis(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tag, eyeID, patID) arrays of shape (488,) like utils.get_diagnosis."""
    path_od = os.path.join(root, "ClinicalData", "patient_data_od.xlsx")
    path_os = os.path.join(root, "ClinicalData", "patient_data_os.xlsx")

    df_od_raw = pd.read_excel(path_od, index_col=[0])
    df_os_raw = pd.read_excel(path_os, index_col=[0])

    df_od = _fix_df(df_od_raw)
    df_os = _fix_df(df_os_raw)

    index_od = np.ones(df_od.iloc[:, 2].values.shape, dtype=np.int8)
    index_os = np.zeros(df_os.iloc[:, 2].values.shape, dtype=np.int8)

    eye_id = np.array(list(zip(index_od, index_os))).reshape(-1)
    tag = np.array(list(zip(df_od.iloc[:, 2].values, df_os.iloc[:, 2].values))).reshape(
        -1
    )
    pat_id = np.array([[int(i.replace("#", ""))] * 2 for i in df_od.index]).reshape(-1)

    return tag, eye_id, pat_id


def main() -> None:
    if not os.path.isdir(FUNDUS_DIR):
        raise SystemExit(f"FundusImages directory not found: {FUNDUS_DIR}")

    # Read diagnosis tags from clinical data
    # y: diagnosis tag array of length 488
    y, eye_id, pat_id = get_diagnosis(PAPILA_ROOT)

    # Collect all image filenames (RETxxxOD/OS.jpg)
    image_files = sorted(
        f
        for f in os.listdir(FUNDUS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if len(image_files) != len(y):
        raise SystemExit(
            f"Mismatch between number of images ({len(image_files)}) "
            f"and diagnosis tags ({len(y)}). Cannot build dataset safely."
        )

    # Map tag -> class name
    def tag_to_class(tag: int) -> str:
        return "healthy" if tag == 0 else "neurological_risk"

    # Build list of (filename, class_name)
    pairs = [(fname, tag_to_class(int(tag))) for fname, tag in zip(image_files, y)]

    # Split into train / val with stratification
    random.seed(RANDOM_SEED)

    by_class: dict[str, list[str]] = {"healthy": [], "neurological_risk": []}
    for fname, cls_name in pairs:
        by_class[cls_name].append(fname)

    splits: dict[str, dict[str, list[str]]] = {
        "train": {"healthy": [], "neurological_risk": []},
        "val": {"healthy": [], "neurological_risk": []},
    }

    for cls_name, files in by_class.items():
        random.shuffle(files)
        split_idx = int(len(files) * TRAIN_SPLIT)
        splits["train"][cls_name] = files[:split_idx]
        splits["val"][cls_name] = files[split_idx:]

    # Reset existing dataset folder
    if os.path.isdir(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)

    for split in ["train", "val"]:
        for cls_name in ["healthy", "neurological_risk"]:
            os.makedirs(os.path.join(DATASET_ROOT, split, cls_name), exist_ok=True)

    # Copy images into new folders
    for split in ["train", "val"]:
        for cls_name, files in splits[split].items():
            dst_dir = os.path.join(DATASET_ROOT, split, cls_name)
            for fname in files:
                src = os.path.join(FUNDUS_DIR, fname)
                dst = os.path.join(dst_dir, fname)
                shutil.copy2(src, dst)

    # Simple summary
    print("Finished building 2-class dataset under 'dataset_papila/'")
    for split in ["train", "val"]:
        for cls_name in ["healthy", "neurological_risk"]:
            count = len(splits[split][cls_name])
            print(f"{split}/{cls_name}: {count} images")


if __name__ == "__main__":
    main()

