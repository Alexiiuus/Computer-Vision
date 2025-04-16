# Image Augmentation Script

This script allows you to apply various image augmentations such as flipping, rotating, cropping, shearing, grayscale conversion, exposure adjustment, and color space transformations to a dataset. It is built using the Albumentations library and can be used to increase the variety of your image data for training machine learning models.

## Features

- Horizontal and vertical flips
- Rotation by specified angles
- Random cropping (percentage-based)
- Shear transformations (both vertical and horizontal)
- Grayscale conversion with probability control
- Exposure adjustment (brightness/contrast modification)
- CMY color filter (inverts colors)
- HSV color space conversion
- Supports datasets with COCO-style annotations

## Requirements

- Python 3.x
- OpenCV (opencv-python)
- Albumentations (albumentations)

You can install the required libraries using pip:

```
pip install opencv-python albumentations
```

## How to Use

### Command-Line Arguments

You can run the script using the following command structure:
```
python generate_augmentations.py --source <path_to_input_dataset> --output <path_to_output_dataset> [options]
```

### Options

- `--source <path>`: The path to the source directory containing the images and annotations.
- `--output <path>`: The path to the output directory where the augmented images and annotations will be saved.
- `--flip-horizontal`: Flip each image horizontally.
- `--flip-vertical`: Flip each image vertically.
- `--rotate <angles>`: List of angles to rotate the images (e.g., --rotate 45 90 180).
- `--crop <percentages>`: List of crop percentages (e.g., --crop 10 20 30).
- `--shear-vertical <angles>`: List of vertical shear angles (e.g., --shear-vertical 10 20).
- `--shear-horizontal <angles>`: List of horizontal shear angles (e.g., --shear-horizontal 10 20).
- `--grayscale <probabilities>`: List of probabilities to convert images to grayscale (e.g., --grayscale 0.5 1.0).
- `--exposure <values>`: List of exposure values (e.g., --exposure 0.2 0.5).
- `--cmy`: Apply CMY color filter (invert colors).
- `--hsv`: Convert image to HSV color space.

## Example Usage

```
python generate_augmentations.py --source dataset/ --output output/ --rotate 45 90 --flip-horizontal --crop 10 20 --grayscale 0.5 1.0 --exposure 0.3 --cmy --hsv
```

This command will:

- Rotate images by 45 and 90 degrees.
- Flip images horizontally.
- Crop images by 10% and 20% from the edges.
- Convert some images to grayscale with a 50% probability and others with 100% probability.
- Adjust the exposure (brightness/contrast) by 30%.
- Apply CMY color inversion.
- Convert images to HSV color space.

## Dataset Support

This script is designed for datasets in the COCO format, where images are associated with annotation files (`annotations.json`). The script ensures that annotations are correctly updated for each transformation applied to the images.

### Example Output

The augmented images and updated annotations will be saved in the specified `--output` folder. The filenames will reflect the transformations applied. For example:

- `image_rotate_45.jpg` for a 45-degree rotation,
- `image_flip_horizontal.jpg` for a horizontal flip,
- `image_crop_10.jpg` for a 10% crop from the edges,
- `image_grayscale_50.jpg` for images converted to grayscale with a 50% probability.

Each transformation will be applied independently, and combinations of transformations will result in multiple augmented images.

## Important Notes

- The dataset should include both images and an `annotations.json` file following the COCO format.
- The script ensures that annotations are updated when transformations affect the image geometry.
- The script applies transformations in sequence, meaning multiple augmentations could be applied to each image.


