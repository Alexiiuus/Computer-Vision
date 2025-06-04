import torch
from ultralytics import YOLO

class YoloModel:
    def __init__(self, model_path, conf=0.5, iou=0.4, task="detect", classes=None, max_det=1000):
        self.model = YOLO(model_path)
        self._data_pred = [conf, iou, task, classes, max_det]
        self.names = self.model.names


    def predict(self, image):
        conf, iou, task, classes, max_det = self._data_pred
        results = self.model.predict(image, conf=conf, iou=iou, task=task, classes=classes, max_det=max_det)[0].boxes
        return (
            results.xyxy.cpu().numpy() if results is not None else [],
            results.conf.cpu().numpy() if results is not None else [],
            results.cls.cpu().numpy() if results is not None else [],
        )

    def cls_name(self, cls):
        return self.names[int(cls)]

class ModelFix(torch.nn.Module):
    def __init__(self, model1, model2, model3, is_model_2= lambda x: True):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.is_model_2 = is_model_2

    def forward(self, image):
        det_1 = self.model1.predict(image)
        det_2 = []
        for box_fish, _, cls_fish in det_1:
            x1, y1, x2, y2 = map(int, box_fish[:4])
            crop = image[y1:y2, x1:x2]
            is_model_2 = self.is_model_2.predict(cls_fish)
            det_2.append(self.model2(crop) if is_model_2 else self.model3(crop))

        # Combinar resultados
        return torch.cat([det_1, det_2], dim=1)
