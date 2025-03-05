import argparse
import os
import cv2
import albumentations as A
import json
from utils import *

def run_augmentations(args, folder_path, output_images_path):
    path_annotation = os.path.join(folder_path, "annotations.json")
    path_images = os.path.join(folder_path, "images")    
    transforms = CrateTranforms(args)
    
    create_folder(output_images_path)
    copy_images(path_images, output_images_path)
    

    if not (exist_all_archives_folders(path_annotation, path_images)): return
  
    data, categories, annotations_by_image = json_data(path_annotation)
    
    data_img = data["images"].copy()
    for id_img, img in enumerate(data_img):
        filepath = os.path.join(path_images, img["file_name"])

        if not os.path.isfile(filepath):
            print("Not found:", filepath)
            continue
        
        image = cv2.imread(filepath)
        if image is None:
            print(f"Error loading image: {filepath}")
            continue

        height, width = img['height'], img['width']
        image_annotations = annotations_by_image.get(img["id"], [])
        bboxes = [ann["bbox"] for ann in image_annotations]
        categories = [ann["category_id"] for ann in image_annotations]
        path_annotation_out = os.path.join(args.output, "annotations.json")
        total_annotation_id, total_image_id, new_images, new_annotations = new_annotations_images(args.output, path_annotation_out, data, img, id_img, annotations_by_image)

        for transform_name, transform_list in transforms:
            for transform in transform_list:  
                if "crop_" in transform_name:
                    crop_percentage = int(transform_name.split("_")[1])
                    crop_height = int(height * (1 - crop_percentage / 100))
                    crop_width = int(width * (1 - crop_percentage / 100))
                    transform = A.CenterCrop(height=crop_height, width=crop_width, p=1.0)

                transform = A.Compose([transform], bbox_params=A.BboxParams(format='coco', min_area=200, label_fields=['category_ids']))
                augmented = transform(image=image, bboxes=bboxes, category_ids=categories)
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']

                base, ext = os.path.splitext(img["file_name"])
                output_name = f"{base.split('/')[-1].rsplit('.', 1)[0]}_{transform_name}{ext}"
                output_path = os.path.join(output_images_path, "images", output_name)
                cv2.imwrite(output_path, augmented_image)
                print(f"Saved: {output_path}")

                new_images.append({
                    "id": total_image_id,
                    "file_name": f"images/{output_name}",
                    "width": augmented_image.shape[1],
                    "height": augmented_image.shape[0]
                })

                for bbox, category in zip(augmented_bboxes, categories):
                    new_annotations.append({
                        "id": total_annotation_id,
                        "image_id": total_image_id,
                        "category_id": category,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    })
                
                    total_annotation_id += 1
                
                total_image_id += 1
        
        if os.path.isfile(path_annotation_out):
            with open(os.path.join(args.output, "annotations.json"), "r") as file:
                existing_data = json.load(file)
            
            existing_data["images"].extend(new_images)
            existing_data["annotations"].extend(new_annotations)

            with open(os.path.join(args.output, "annotations.json"), "w") as file:
                json.dump(existing_data, file, indent=4)
        else:
            data_modify = {
                "images": new_images,
                "annotations": new_annotations,
                "categories": data["categories"]
            } 
            
            with open(os.path.join(args.output, "annotations.json"), "w") as file:
                json.dump(data_modify, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output", type=str, default="output", required=True, help="Path to the output folder")
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

    # if os.path.isdir(args.source):
    #     path_annotation = os.path.join(args.source, "annotations.json")
    #     if os.path.isfile(path_annotation):
    #         folders = [args.source]
    #     else:
    #         folders = [ os.path.join(args.source, f) for f in sorted(os.listdir(args.source)) ]
    # else:
    #     print(f"Not found: {args.source}")

    # # Procesar todos los videos en la carpeta de entrada
    # for folder_path in folders:
    #     if not os.path.isdir(folder_path):
    #         print(f"It's not a folder: {folder_path}")
    #         continue
        
    #     print(f"Processing: {folder_path}")
        
    #     run_augmentations(args, folder_path, args.output)