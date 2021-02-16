import torch
from PIL import Image

class Detector(object):
    def __init__(self):
        self.model_types = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]
    
    def load_model(self, model_name, pretrained):
        if model_name not in self.model_types:
            raise("Model Type not Found !")
        else:
            return torch.hub.load("ultralytics/yolov5", model_name, pretrained)

    def make_predictions(self, model, img_name):
        # Make sure that the that Image is represented as RGB
        image = Image.open(img_name)
        return model(image)


