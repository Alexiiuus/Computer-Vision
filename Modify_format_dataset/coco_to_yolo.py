import json
import os
import argparse

def coco_to_yolo(dataset_path, output_path):
    # Rutas
    images_dir = os.path.join(dataset_path, "images")
    annotations_path = os.path.join(dataset_path, "annotations.json")
    labels_dir = os.path.join(output_path, "labels")

    # Crear carpeta de salida si no existe
    os.makedirs(labels_dir, exist_ok=True)

    # Cargar anotaciones COCO
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Mapeo de categorías
    category_map = {cat["id"]: i for i, cat in enumerate(coco_data["categories"])}

    # Diccionario de imágenes
    images_info = {img["id"]: img for img in coco_data["images"]}

    # Procesar anotaciones
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        img_info = images_info[img_id]
        width, height = img_info["width"], img_info["height"]

        # Convertir bbox a formato YOLO
        x_min, y_min, w, h = ann["bbox"]
        x_center = (x_min + w / 2) / width
        y_center = (y_min + h / 2) / height
        w /= width
        h /= height
        class_id = category_map[ann["category_id"]]

        # Nombre del archivo .txt
        txt_filename = os.path.splitext(img_info["file_name"])[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_filename)

        # Crear el directorio si no existe
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        # Guardar en formato YOLO
        with open(txt_path, "a") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


    print(f"Conversión completa. Archivos YOLO guardados en: {labels_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output", type=str, default="output", required=True, help="Path to the output folder")
    
    args = parser.parse_args()
    
    coco_to_yolo(args.source, args.output)
