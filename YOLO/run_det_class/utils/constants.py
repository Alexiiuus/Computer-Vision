# -- Constants for the models
FOLDER_MODELS_PATH = "weights"
DETECTION_MODEL_PATH = f"{FOLDER_MODELS_PATH}/yolov11x_rice-detectionV15.pt"
CLASSIFICATION_MODEL_PATH = f"{FOLDER_MODELS_PATH}/yolov11x_rice-classificationV6.pt"
SEGMENTATION_CLASS_MODEL_PATH = f"{FOLDER_MODELS_PATH}/rice-classification-yolo11mV2.pt"
CONF_DETECTION = 0.1
CONF_CLASSIFICATION = 0.5
CONF_IOU = 0.45
# -- Constants for the videos
FOLDER_VIDEOS_PATH = "input/videos_prub"

# -- Constants for the class
FILES_VALIDS = ('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.png')
CROP_SIZE_IMAGE = 128