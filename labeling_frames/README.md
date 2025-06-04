# Labeling Frames with Bounding Boxes

This tool allows you to interactively draw bounding boxes on an image or a video frame using your mouse, assign class labels to them, and save the annotations in COCO format (as a JSON file).

## Features

* Draw bounding boxes with the mouse.
* Assign numeric class labels (0–9) to each bounding box.
* Visual feedback of the bounding boxes and class input.
* Supports both images and video files.
* Saves annotations in COCO-compatible format.

## Requirements

* Python 3.7+
* OpenCV (`cv2`)
* NumPy
* [supervision](https://github.com/roboflow/supervision)

Install dependencies with:

```bash
pip install opencv-python numpy supervision
```

## Usage

```bash
python labeling_frames.py --source_path path/to/image_or_video --bbox_configuration_path path/to/output.json
```

### Arguments

* `--source_path` (required): Path to the input image or video file.
* `--bbox_configuration_path`: Path to the output annotation JSON file. Default is `out.json`.

## Controls

| Key     | Action                                            |
| ------- | ------------------------------------------------- |
| Mouse   | Click and drag to draw a bounding box             |
| 0–9     | Input class label digits                          |
| `c`     | Clear the current class input                     |
| `Enter` | Confirm the class label for the last bounding box |
| `s`     | Save the annotations to the output JSON file      |
| `q`     | Quit the program                                  |

## Output Format

The annotations are saved in **COCO format**, with the following fields:

* `images`: Metadata about the image (file name, dimensions, ID)
* `annotations`: List of bounding boxes with coordinates, category ID, and area
* `categories`: Class definitions with numeric IDs and names (`class_0`, `class_1`, etc.)

## Example

```bash
python labeling_frames.py --source_path data/frame.jpg --bbox_configuration_path labels.json
```

## Notes

* The image is automatically resized to 50% of its original size for easier handling.
* If the output JSON file already exists, new annotations will be appended while avoiding ID collisions.
