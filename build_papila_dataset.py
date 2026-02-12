import os
import shutil
import random

# Paths relative to the project root (where train.py lives)
PAPILA_FUNDUS_DIR = os.path.join(
    "PapilaDB-PAPILA-17f8fa7746adb20275b5b6a0d99dc9dfe3007e9f",
    "FundusImages",
)
DATASET_ROOT = "dataset"

# We'll create a single class folder; the model will still run
CLASS_NAME = "papila"
TRAIN_SPLIT = 0.8  # 80% train, 20% val


def main() -> None:
    if not os.path.isdir(PAPILA_FUNDUS_DIR):
        raise SystemExit(f"Fundus image directory not found: {PAPILA_FUNDUS_DIR}")

    # Collect all image filenames
    images = [
        f
        for f in os.listdir(PAPILA_FUNDUS_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not images:
        raise SystemExit(f"No images found in {PAPILA_FUNDUS_DIR}")

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_SPLIT)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_dir = os.path.join(DATASET_ROOT, "train", CLASS_NAME)
    val_dir = os.path.join(DATASET_ROOT, "val", CLASS_NAME)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _copy(img_list, dst_root):
        for name in img_list:
            src = os.path.join(PAPILA_FUNDUS_DIR, name)
            dst = os.path.join(dst_root, name)
            shutil.copy2(src, dst)

    _copy(train_images, train_dir)
    _copy(val_images, val_dir)

    print(f"Created dataset under '{DATASET_ROOT}':")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images:   {len(val_images)}")
    print(f"  Class name:   {CLASS_NAME}")


if __name__ == "__main__":
    main()

