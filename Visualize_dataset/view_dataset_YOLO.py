import os
import cv2
import argparse

def draw_custom_bbox_format(image, label_path, color=(0, 255, 0), thickness=2):
    h, w = image.shape[:2]
    drawn = False

    if not os.path.exists(label_path):
        return image, drawn

    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) == 11:
                try:
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    xs = coords[0::2]
                    ys = coords[1::2]

                    xmin = int(min(xs) * w)
                    ymin = int(min(ys) * h)
                    xmax = int(max(xs) * w)
                    ymax = int(max(ys) * h)

                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
                    cv2.putText(image, str(cls_id), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    drawn = True
                except Exception as e:
                    print(f"Error procesando línea en {label_path}: {e}")
                    continue

    return image, drawn

def save_visualizations(dataset_path, output_path, max_images=100):
    image_dir = os.path.join(dataset_path, "images")
    label_dir = os.path.join(dataset_path, "labels")

    os.makedirs(output_path, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    saved_count = 0

    for img_name in sorted(image_files):
        if saved_count >= max_images:
            break

        image_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {img_name}")
            continue

        image_with_boxes, has_detections = draw_custom_bbox_format(image, label_path)

        if has_detections:
            output_img_path = os.path.join(output_path, img_name)
            cv2.imwrite(output_img_path, image_with_boxes)
            print(f"[✓] Guardada con detecciones: {output_img_path}")
            saved_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Ruta al dataset YOLO (con /images y /labels)")
    parser.add_argument("--output", type=str, default="visualized", help="Carpeta de salida para las imágenes visualizadas")
    parser.add_argument("--max", type=int, default=100, help="Cantidad máxima de imágenes a guardar")
    args = parser.parse_args()

    save_visualizations(args.dataset, args.output, args.max)
