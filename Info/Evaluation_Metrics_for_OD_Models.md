# Evaluation Metrics for Object Detection Models (YOLO)

This repository aims to document and explain the most common metrics used to evaluate object detection models, particularly those based on architectures like YOLO (You Only Look Once).

## 📊 mAP (mean Average Precision)

**mAP** is one of the most important and widely used metrics for assessing the performance of object detection models.

### What is mAP?

**mAP (mean Average Precision)** measures the average precision of a model based on different Intersection over Union (**IoU**) thresholds between predicted bounding boxes and the ground truth.

In simple terms, it evaluates:
- **Precision**: How accurate the predictions are.
- **Recall**: How many of the actual objects were detected.

mAP combines both by computing the area under the Precision-Recall curve for each class and then averaging over all classes.

### How is it calculated?

1. Compute the **Average Precision (AP)** for each class by plotting the Precision-Recall curve at a specific IoU threshold (e.g., 0.5).
2. The mean of these AP values across all classes gives the **mAP**.

Examples:
- **mAP@0.5**: Average Precision at IoU = 0.5.
- **mAP@0.5:0.95**: Average Precision averaged over multiple IoU thresholds (from 0.5 to 0.95 in steps of 0.05), offering a more comprehensive and stricter evaluation.

### Why is it important?

A high mAP means the model is good at detecting and accurately localizing objects. It is essential for comparing the performance between different model versions or across different models.

Useful for: Tumor detection, autonomous vehicles, etc.

### Reference

- [Ultralytics Glossary: mAP](https://www.ultralytics.com/es/glossary/mean-average-precision-map)

