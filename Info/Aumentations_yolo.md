
# 🧪 YOLOv8 (and others) Data Augmentation Guide

This README explains how data augmentation works in [Ultralytics YOLO](https://docs.ultralytics.com/guides/yolo-data-augmentation/) and how to use its configurable parameters during training. These augmentations help improve generalization by presenting the model with a wider variety of training data.

## 🔍 What is On-the-Fly Data Augmentation?

Data augmentations in Ultralytics YOLO are applied **dynamically during training** — images are transformed randomly **each time they're loaded into a batch**. This means:
- No changes are made to the dataset files on disk.
- Every training epoch may apply different transformations.
- It increases the diversity of input data **without increasing disk usage**.

---

## ⚙️ How to Configure Augmentations

You can customize augmentations by passing a `cfg` dictionary to the `train()` function or by modifying your `*.yaml` config file.

Example (Python):

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="data.yaml", epochs=100, augment=True, 
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
            perspective=0.0, flipud=0.0, fliplr=0.5,
            mosaic=1.0, mixup=0.0, copy_paste=0.0)
```

---

## 🧾 Parameter Descriptions

| Parameter     | Type    | Description |
|---------------|---------|-------------|
| `hsv_h`       | float   | Image hue augmentation (0.0 to 1.0). Slight color shifts. |
| `hsv_s`       | float   | Saturation augmentation (0.0 to 1.0). Adjusts vividness of color. |
| `hsv_v`       | float   | Value (brightness) augmentation (0.0 to 1.0). |
| `degrees`     | float   | Rotation angle range (±degrees). |
| `translate`   | float   | Image translation as a fraction of image size (0.0 to 1.0). |
| `scale`       | float   | Scaling factor (e.g., 0.5 = 50% scale variation). |
| `shear`       | float   | Shear angle (±degrees). Slants the image. |
| `perspective` | float   | Perspective transformation. Adds depth-like distortion. |
| `flipud`      | float   | Probability of vertical flip (0.0 to 1.0). |
| `fliplr`      | float   | Probability of horizontal flip. Very common. |
| `mosaic`      | float   | Probability of Mosaic augmentation (combines 4 images). |
| `mixup`       | float   | Probability of MixUp (blends two images/labels). |
| `copy_paste`  | float   | Probability of Copy-Paste (paste objects between images). |

---

## 📌 Tips for Usage

- **Default values** in YOLOv8 are usually well-balanced for general use.
- If you're training on a small dataset, increasing `mosaic`, `mixup`, and color augmentations may help.
- For highly structured datasets (e.g., OCR), disable perspective, flip, etc.
- Run training with `augment=False` if you want to test performance without augmentations.

---

## 📦 Need to Save Augmented Images?

Ultralytics augmentations are not saved to disk. To **generate and store augmented images**, use external libraries like:
- [Albumentations](https://albumentations.ai/)
- [Roboflow](https://roboflow.com/)
- Custom Python scripts

---

## 📚 Resources

- [Official Ultralytics Docs - Data Augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- [YOLOv8 GitHub Repo](https://github.com/ultralytics/ultralytics)