import os
import albumentations as A
import json
import shutil
import cv2

def cmy_transform(image, **kwargs):
    return 255 - image  # Inversión de colores RGB para simular CMY

def hsv_transform(image, **kwargs):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def CrateTranforms(args):
    transforms = []
    if args.flip_horizontal:
        transforms.append(("horizontal", [A.HorizontalFlip(p=1.0)]))
    if args.flip_vertical:
        transforms.append(("vertical", [A.VerticalFlip(p=1.0)]))
    if args.rotate:
        angles = [float(angle) for angle in args.rotate]
        for angle in angles:
            transforms.append((f"rotate_{int(angle)}", [A.Rotate(limit=(angle, angle), p=1.0)]))
    if args.crop:
        crop_percentages = [float(crop) for crop in args.crop]
        for crop_percentage in crop_percentages:
            if 0 < crop_percentage < 100:
                transforms.append((f"crop_{int(crop_percentage)}", [None]))
            else:
                print(f"Error: Crop percentage {crop_percentage} must be between 0 and 100.")
                return []
    if args.shear_vertical:
        shears_vertical = [float(shear) for shear in args.shear_vertical]
        for shear in shears_vertical:
            transforms.append((f"shear_vertical_{int(shear)}", [A.Affine(shear={'x': 0, 'y': shear}, p=1.0)]))
    if args.shear_horizontal:
        shears_horizontal = [float(shear) for shear in args.shear_horizontal]
        for shear in shears_horizontal:
            transforms.append((f"shear_horizontal_{int(shear)}", [A.Affine(shear={'x': shear, 'y': 0}, p=1.0)]))
    if args.grayscale:
        grayscale_probabilities = [float(prob) for prob in args.grayscale]
        for prob in grayscale_probabilities:
            if 0 <= prob <= 1:
                transforms.append((f"grayscale_{int(prob*100)}", [A.ToGray(p=prob)]))
            else:
                print(f"Error: Grayscale probability {prob} must be between 0 and 1.")
                return []
    if args.exposure:
        exposure_values = [float(exp) for exp in args.exposure]
        for exp in exposure_values:
            if 0 <= exp <= 1:
                transforms.append((f"exposure_{int(exp*100)}", [A.RandomBrightnessContrast(brightness_limit=exp, p=1.0)]))
            else:
                print(f"Error: Exposure value {exp} must be between 0 and 1.")
                return []
    if args.cmy:
        transforms.append(("cmy", [A.Lambda(image=cmy_transform)]))  # Inversión de colores RGB para simular CMY
    if args.hsv:
        transforms.append(("hsv", [A.Lambda(image=hsv_transform)]))
    
    return transforms if transforms else []


def exist_all_archives_folders(path_annotation, path_images):
    if not (os.path.isfile(path_annotation) or os.path.isdir(path_images)):
        print(f"Not found: {path_annotation} or {path_images}")
        return None

    return True

def create_folder(output_images_path):
    os.makedirs(output_images_path, exist_ok= True)
    os.makedirs(f"{output_images_path}/images", exist_ok= True)

def copy_images(path_images, output_images_path):
    for filename in os.listdir(path_images):
        source_path = os.path.join(path_images, filename)
        dest_path = os.path.join(output_images_path, "images", f"{filename}")
        if os.path.isfile(source_path):
            shutil.copy(source_path, dest_path)

def json_data(path_annotation):
    with open(path_annotation, "r") as file:
        data = json.load(file)

    categories = data['categories']
    annotations_by_image = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    #print("Claves principales del JSON:", list(data.keys())) 
    return data, categories, annotations_by_image

def total_ids_ann_img(output_path, path_annotation_out):
    total_annotation_id = 0
    total_image_id = 0

    if os.path.isfile(path_annotation_out):
        with open(os.path.join(output_path, "annotations.json"), "r") as file:
            existing_data = json.load(file)
        total_annotation_id = len(existing_data["annotations"])
        total_image_id = len(existing_data["images"])
    
    return total_annotation_id, total_image_id

def new_annotations_images(output_path, path_annotation_out, data, img, id_img, annotations_by_image):
    total_annotation_id, total_image_id = total_ids_ann_img(output_path, path_annotation_out)
    new_images = []
    new_annotations = []
    data["images"][id_img]['file_name'] = os.path.join("images", img['file_name'])
    data["images"][id_img]['id'] = total_image_id
    new_images.append(data["images"][id_img])
    annotations_i = annotations_by_image.get(id_img, [])
    
    for ann in annotations_i:
        new_annotations.append({
                "id": total_annotation_id,
                "image_id": total_image_id,
                "category_id": ann['category_id'],
                "bbox": ann['bbox'],
                "area": ann['bbox'][2] * ann['bbox'][3],
                "iscrowd": 0
            })
        total_annotation_id += 1
    total_image_id = total_image_id + len(new_images)

    return total_annotation_id, total_image_id, new_images, new_annotations
