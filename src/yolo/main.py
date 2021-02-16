import torch
from PIL import Image
# from detector import Detector_Class
from detector import Detector 
from pprint import pprint

def main():
    Model = Detector()
    model = Model.load_model("yolov5s", pretrained = True)

    path = "../data/images/"
    images = ["bus.jpg", "zidane.jpg"]

    for image_name in images:
        filename = path + image_name
        predictions = Model.make_predictions(model, filename)

        predictions.print()
        predictions.save()

main()