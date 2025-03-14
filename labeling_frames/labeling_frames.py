import argparse
import json
import os
from typing import Any, Optional, Tuple
import cv2
import numpy as np
import random  # Para generar colores aleatorios

import supervision as sv

KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
KEY_CTRL_Q = 17  # Detect Ctrl key

THICKNESS = 2
WINDOW_NAME = "Draw Bounding Boxes"
BBOXES = []

current_mouse_position: Optional[Tuple[int, int]] = None
current_bbox_start: Optional[Tuple[int, int]] = None
current_class_input: str = ""

# Diccionario para almacenar colores por clase
class_colors = {}

def resolve_source(source_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(source_path):
        return None

    image = cv2.imread(source_path)
    if image is not None:
        return image

    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    frame = next(frame_generator)
    return frame

def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global current_mouse_position, current_bbox_start, current_class_input
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        current_bbox_start = (x, y)
        
    elif event == cv2.EVENT_LBUTTONUP and current_bbox_start is not None:
        x1, y1 = current_bbox_start
        x2, y2 = x, y
        if abs(x2 - x1) > 0 and abs(y2 - y1) > 0:
            BBOXES.append([min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), ""])
        
        current_bbox_start = None

def generate_class_color(cls_id: str) -> Tuple[int, int, int]:
    """Genera un color único para cada clase. Si ya existe un color, lo devuelve."""
    if cls_id not in class_colors:
        # Genera un color aleatorio si no existe un color para esta clase
        # Usamos un color basado en el ID de la clase para asegurarnos de que sea único.
        class_colors[cls_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return class_colors[cls_id]


def draw_class_input(image: np.ndarray) -> None:
    """Dibuja el campo de entrada de la clase sobre la imagen"""
    global current_class_input
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Class: {current_class_input}"
    cv2.putText(image, text, (10, image.shape[0] - 10), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
    global current_mouse_position, current_bbox_start
    image[:] = original_image.copy()

    # Dibuja las cajas delimitadoras existentes
    for bbox in BBOXES:
        x, y, w, h, cls_id = bbox
        color = generate_class_color(cls_id)  # Obtener el color de la clase
        cv2.rectangle(image, (x, y), (x + w, y + h), color, THICKNESS)

    # Dibuja la caja actual si está siendo arrastrada
    if current_bbox_start and current_mouse_position:
        x1, y1 = current_bbox_start
        x2, y2 = current_mouse_position
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), THICKNESS)

    # Dibuja el campo de texto para la clase
    draw_class_input(image)

    cv2.imshow(WINDOW_NAME, image)

def save_bboxes_to_coco(bboxes, target_path, image_id=1, image_filename="image.jpg", image_width=1920, image_height=1080):
    # Cargar las anotaciones existentes si el archivo ya existe
    if os.path.exists(target_path):
        with open(target_path, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {
            "images": [],
            "annotations": [],
            "categories": []
        }

    # Obtener el ID máximo de la categoría y las anotaciones para evitar colisiones de ID
    existing_category_ids = {cat["id"] for cat in annotations["categories"]}
    existing_annotation_ids = {ann["id"] for ann in annotations["annotations"]}

    print(bboxes)
    unique_classes = sorted(set(int(bbox[4]) for bbox in bboxes if len(bbox) == 5 and bbox != ""))
    current_annotation_id = max(existing_annotation_ids, default=0) + 1
    current_category_id = max(existing_category_ids, default=0) + 1

    # Actualizar las categorías si es necesario
    for cls_id in unique_classes:
        if cls_id not in existing_category_ids:
            annotations["categories"].append({"id": current_category_id, "name": f"class_{cls_id}"})
            current_category_id += 1

    # Añadir la nueva imagen
    annotations["images"].append({
        "id": image_id,
        "file_name": image_filename,
        "width": image_width,
        "height": image_height
    })

    # Añadir las nuevas anotaciones
    for bbox in bboxes:
        if len(bbox) == 5:
            annotations["annotations"].append({
                "id": current_annotation_id,
                "image_id": image_id,
                "bbox": bbox[:4],
                "category_id": int(bbox[4]),
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            current_annotation_id += 1

    # Guardar el archivo con las nuevas anotaciones
    with open(target_path, "w") as f:
        json.dump(annotations, f, indent=4)

def main(source_path: str, bbox_configuration_path: str) -> None:
    global current_mouse_position, current_class_input
    original_image = resolve_source(source_path=source_path)
    if original_image is None:
        print("Failed to load source image.")
        return

    image_filename = os.path.basename(source_path)
    image_height, image_width = original_image.shape[:2]
    
    # Reducir el tamaño de la imagen al 50%
    resized_image = cv2.resize(original_image, (image_width // 2, image_height // 2))
    
    image = resized_image.copy()
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Cambia el modo de entrada de clase si presionas 'c'
        if key == ord("c"):
            current_class_input = ""  # Reinicia la clase

        # Si se presiona una tecla, actualizar el campo de clase
        elif 48 <= key <= 57:  # Si se presiona un número (0-9)
            current_class_input += chr(key)  # Añadir el número a la clase
        
        # Para finalizar el ingreso de la clase y aplicar el rectángulo con ella
        elif key == KEY_ENTER:
            if current_class_input:
                BBOXES[-1][-1] = current_class_input
                print(f"Class set to: {current_class_input}")
                current_class_input = ""
        # Guardar las anotaciones cuando se presiona 's'
        elif key == KEY_SAVE:
            save_bboxes_to_coco(BBOXES, bbox_configuration_path, image_filename=image_filename, image_width=image_width, image_height=image_height)
            print(f"Bounding boxes saved to {bbox_configuration_path}")
        
        # Salir cuando se presiona 'q'
        elif key == ord("q"):
            print("Exiting...")
            break
        
        redraw(image, resized_image)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively draw bounding boxes on images or video frames and save as COCO format."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        required=True,
        help="Path to the source image or video file for drawing bounding boxes.",
    )
    parser.add_argument(
        "--bbox_configuration_path",
        type=str,
        required=True,
        help="Path where the bounding box annotations will be saved as a JSON file.",
    )
    arguments = parser.parse_args()
    main(
        source_path=arguments.source_path,
        bbox_configuration_path=arguments.bbox_configuration_path,
    )
