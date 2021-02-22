import torch
from PIL import Image

class Detector(object):
    def __init__(self):
        self.model_types = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]
    
    def load_model(self, model_name, pretrained, n_classes = 80):
        if model_name not in self.model_types:
            raise Exception(f"Model - {model_name} not supported!")
        else:
            return torch.hub.load("ultralytics/yolov5", model_name, pretrained, classes = n_classes)

    def make_general_predictions(self, model, img_name):
        # Make sure that the that Image is represented as RGB
        '''
        @param [model] - model that it uses, see self.model_types for details
        @param [img_name] - filename of the image that it uses
        @returns - the boxes of all possible 80 classes that the model has been trained on 
        '''
        image = Image.open(img_name)
        return model(image)

    def predict_people(self, model, img_name, classes = [0]):
        '''
        @param [model] - model that it uses, see self.model_types for details
        @param [img_name] - filename of the image that it uses
        @returns - the box that of only Human Beings
        '''
        # Check the autoShape class for this param
        model.classes = classes
        image = Image.open(img_name)
        return model(image)

        

        



