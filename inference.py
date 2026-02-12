import os

import torch
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "dataset_papila"

val_transforms = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
classes = train_dataset.classes


if len(classes) < 2:
    raise SystemExit(
        "Your training data only has one class: "
        f"{classes}.\\n"
        "To get BOTH 'Neurological Disorder Risk Detected' and "
        "'No Neurological Disorder Detected' outputs with meaningful "
        "confidence and accuracy, you must organise your data as:\n"
        "  dataset/train/healthy\n"
        "  dataset/train/neurological_risk\n"
        "  dataset/val/healthy\n"
        "  dataset/val/neurological_risk\n"
        "and then re-run train.py to retrain the model."
    )

model = models.efficientnet_b3(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = val_transforms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)

    idx = pred_idx.item()
    label = classes[idx]
    conf_percent = conf.item() * 100.0
    return idx, label, conf_percent


def describe_prediction(filename, label, confidence):
    """
    Print result in a user-friendly form like:

    Image: eye2 (44).jpg
    Neurological Disorder Risk Detected
    Confidence: 87.93 %
    """
    print(f"Image: {filename}")

    # Heuristic: if class name sounds healthy/normal, treat as no risk.
    lower_label = label.lower()
    is_no_risk = any(
        key in lower_label for key in ["healthy", "normal", "no_dr", "no_neuro"]
    )

    if is_no_risk:
        print("No Neurological Disorder Detected")
    else:
        print("Neurological Disorder Risk Detected")

    print(f"Confidence: {confidence:.2f} %\n")


def evaluate_on_validation():
    """
    Run the trained model on the validation set and:
    - print result for each image (risk / no risk + confidence)
    - print overall classification accuracy in %

    This expects your folders to look like:
      dataset/val/healthy
      dataset/val/neurological_risk
    (or similar names; 'healthy'/'normal' will be treated as no risk).
    """
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"), transform=val_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0

    for (images, labels), (path, _) in zip(val_loader, val_dataset.samples):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

        pred_idx = preds.item()
        true_idx = labels.item()
        conf_percent = confs.item() * 100.0
        pred_label = classes[pred_idx]

        filename = os.path.basename(path)
        describe_prediction(filename, pred_label, conf_percent)

        if pred_idx == true_idx:
            correct += 1
        total += 1

    if total > 0:
        acc = correct / total * 100.0
        print(f"Overall validation accuracy: {acc:.2f} %")
    else:
        print("No images found in dataset/val.")


if __name__ == "__main__":
    evaluate_on_validation()
