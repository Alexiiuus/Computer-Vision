import argparse
import os
import cv2
import albumentations as A
import json
from utils import *
import pandas as pd

def run_augmentations(args, folder_path, output_images_path):
    path_annotation = os.path.join(folder_path, "_annotations.csv")
    path_images = os.path.join(folder_path)    
    transforms = CrateTranforms(args)
    
    if not os.path.exists(path_annotation):
        print(f"No se encontró el archivo CSV en {path_annotation}")
        return

    df = pd.read_csv(path_annotation)
    
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path, exist_ok=True)

    augmented_rows = []

    for index, row in df.iterrows():
        filename, width, height, cls, xmin, ymin, xmax, ymax = row
        filepath = os.path.join(path_images, filename)
        if not os.path.isfile(filepath):
            print(f"Archivo no encontrado: {filepath}")
            continue

        image = cv2.imread(filepath)
        if image is None:
            print(f"Error al leer la imagen: {filepath}")
            continue

        bbox = [xmin, ymin, xmax, ymax]

        for transform_name, transform_list in transforms:
            for transform in transform_list:
                # Para crop dinámico
                if "crop_" in transform_name:
                    crop_percentage = int(transform_name.split("_")[1])
                    crop_height = int(image.shape[0] * (1 - crop_percentage / 100))
                    crop_width = int(image.shape[1] * (1 - crop_percentage / 100))
                    transform = A.Compose(
                        [A.CenterCrop(height=crop_height, width=crop_width, p=1.0)],
                        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
                    )
                else:
                    transform = A.Compose(
                        [transform],
                        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
                    )

                try:
                    transformed = transform(image=image, bboxes=[bbox], category_ids=[cls])
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']

                    if not transformed_bboxes:
                        continue  # La caja quedó fuera del área visible

                    new_filename = f"{os.path.splitext(filename)[0]}_{transform_name}.jpg"
                    output_filepath = os.path.join(output_images_path, new_filename)
                    cv2.imwrite(output_filepath, transformed_image)

                    x_min, y_min, x_max, y_max = transformed_bboxes[0]
                    augmented_rows.append({
                        'filename': new_filename,
                        'width': transformed_image.shape[1],
                        'height': transformed_image.shape[0],
                        'class': cls,
                        'xmin': int(x_min),
                        'ymin': int(y_min),
                        'xmax': int(x_max),
                        'ymax': int(y_max)
                    })
                except Exception as e:
                    print(f"Error aplicando {transform_name} a {filename}: {e}")

    if augmented_rows:
        output_csv_path = os.path.join(output_images_path, "_annotations.csv")
        df_augmented = pd.DataFrame(augmented_rows)
        df_augmented.to_csv(output_csv_path, index=False)
        print(f"Se guardaron las anotaciones aumentadas en {output_csv_path}")
    else:
        print("No se generaron datos aumentados.")

                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output", type=str, default="output", help="Path to the output folder")
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