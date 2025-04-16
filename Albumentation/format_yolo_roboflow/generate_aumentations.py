import argparse
import os
import cv2
import albumentations as A
from utils import CrateTranforms  # Asegúrate de tener esta función
import glob

def yolo_to_voc(x_center, y_center, width, height, img_w, img_h):
    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2
    return [xmin, ymin, xmax, ymax]

def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return [x_center, y_center, width, height]

def run_augmentations(args, folder_path, output_path):
    images_path = os.path.join(folder_path, "images")
    labels_path = os.path.join(folder_path, "labels")

    output_images = os.path.join(output_path, "images")
    output_labels = os.path.join(output_path, "labels")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    transforms = CrateTranforms(args)

    image_files = glob.glob(os.path.join(images_path, "*.jpg")) + glob.glob(os.path.join(images_path, "*.png"))

    for img_file in image_files:
        img_filename = os.path.basename(img_file)
        label_file = os.path.join(labels_path, os.path.splitext(img_filename)[0] + ".txt")

        image = cv2.imread(img_file)
        if image is None:
            print(f"No se pudo leer la imagen {img_filename}")
            continue

        img_h, img_w = image.shape[:2]

        bboxes = []
        classes = []

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 11:    
                        cls_id = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        xs = coords[0::2]
                        ys = coords[1::2]

                        xmin = min(xs)
                        ymin = min(ys)
                        xmax = max(xs)
                        ymax = max(ys)

                        bbox = [xmin * img_w, ymin * img_h, xmax * img_w, ymax * img_h]
                        bboxes.append(bbox)
                        classes.append(cls_id)
        else:
            print(f"Archivo de etiquetas no encontrado para {img_filename}. Se aplicarán aumentos sin bboxes.")

        for transform_name, transform_list in transforms:
            for transform in transform_list:
                if "crop_" in transform_name:
                    crop_percentage = int(transform_name.split("_")[1])
                    crop_height = int(img_h * (1 - crop_percentage / 100))
                    crop_width = int(img_w * (1 - crop_percentage / 100))
                    transform = A.Compose(
                        [A.CenterCrop(height=crop_height, width=crop_width, p=1.0)],
                        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']) if bboxes else None
                    )
                else:
                    transform = A.Compose(
                        [transform],
                        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']) if bboxes else None
                    )

                try:
                    if bboxes:
                        transformed = transform(image=image, bboxes=bboxes, category_ids=classes)
                        transformed_bboxes = transformed['bboxes']
                        transformed_classes = transformed['category_ids']
                    else:
                        transformed = transform(image=image)
                        transformed_bboxes = []
                        transformed_classes = []

                    transformed_image = transformed['image']
                    new_filename = f"{os.path.splitext(img_filename)[0]}_{transform_name}.jpg"
                    new_labelname = f"{os.path.splitext(img_filename)[0]}_{transform_name}.txt"

                    cv2.imwrite(os.path.join(output_images, new_filename), transformed_image)

                    if transformed_bboxes:
                        new_h, new_w = transformed_image.shape[:2]
                        with open(os.path.join(output_labels, new_labelname), 'w') as f:
                            for bbox, cls in zip(transformed_bboxes, transformed_classes):
                                x_min, y_min, x_max, y_max = bbox

                                # Definir los 4 puntos del rectángulo
                                points = [
                                    (x_min, y_min),  # x0, y0
                                    (x_max, y_min),  # x1, y1
                                    (x_max, y_max),  # x2, y2
                                    (x_min, y_max),  # x3, y3
                                    (x_min, y_min),  # x0, y0 repetido
                                ]

                                # Normalizar
                                norm_points = [(x / new_w, y / new_h) for x, y in points]

                                # Escribir en formato: class_id x0 y0 x1 y1 ... x0 y0
                                flat = [f"{p:.6f}" for point in norm_points for p in point]
                                f.write(f"{int(cls)} {' '.join(flat)}\n")

                    else:
                        with open(os.path.join(output_labels, new_labelname), 'w') as f:
                            f.write("")

                except Exception as e:
                    print(f"Error aplicando {transform_name} a {img_filename}: {e}")

    print(f"Aumentaciones completadas. Salida: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output", type=str, default="output1", help="Path to the output folder")
    parser.add_argument("--flip-horizontal", action="store_true", help="Flip each image horizontally")
    parser.add_argument("--flip-vertical", action="store_true", help="Flip each image vertically")
    parser.add_argument("--rotate", nargs="*", default=[], help="List of degrees to rotate (e.g., 45 90 180)")
    parser.add_argument("--crop", nargs="*", default=[], help="List of percentages to crop from the edges towards the center (e.g., 10 20 30)")
    parser.add_argument("--shear-vertical", nargs="*", default=[], help="List of shear values in degrees for vertical shearing (e.g., 10 20 30)")
    parser.add_argument("--shear-horizontal", nargs="*", default=[], help="List of shear values in degrees for horizontal shearing (e.g., 10 20 30)")
    parser.add_argument("--grayscale", nargs="*", default=[], help="List of probabilities to convert to grayscale (e.g., 0.5 1.0)")
    parser.add_argument("--exposure", nargs="*", default=[], help="List of exposure values (e.g., 0.2 0.5)")
    parser.add_argument("--cmy", action="store_true", help="Apply CMY color filter (invert colors)")
    parser.add_argument("--hsv", action="store_true", help="Convert image to HSV color space")

    args = parser.parse_args()
    run_augmentations(args, args.source, args.output)