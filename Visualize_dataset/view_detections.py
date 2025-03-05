import os
import cv2
import json
import argparse

def draw_bboxes(image, bboxes, categories):
    for bbox, category in zip(bboxes, categories):
        x, y, w, h = map(int, bbox)  # Convertir a enteros
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(category), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def visualize_annotations(annotation_path, image_dir, output_dir):
    if not os.path.isfile(annotation_path):
        print(f"Error: No se encontró {annotation_path}")
        return
    
    with open(annotation_path, "r") as file:
        data = json.load(file)
    
    annotations_by_image = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img in data["images"]:
        file_path = os.path.join(image_dir, img["file_name"])
        if not os.path.isfile(file_path):
            print(f"No existe: {file_path}")
            continue
        
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error al cargar la imagen: {file_path}")
            continue
        
        image_annotations = annotations_by_image.get(img["id"], [])
        bboxes = [ann["bbox"] for ann in image_annotations]
        categories = [ann["category_id"] for ann in image_annotations]
        
        image = draw_bboxes(image, bboxes, categories)
        output_path = os.path.join(output_dir, img["file_name"])
        cv2.imwrite(output_path, image)
        print(f"Guardado: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, required=True, help="Path al archivo annotations.json")
    parser.add_argument("--images", type=str, required=True, help="Path al directorio de imágenes")
    parser.add_argument("--output", type=str, required=True, help="Path al directorio de salida")
    args = parser.parse_args()
    
    visualize_annotations(args.annotations, args.images, args.output)