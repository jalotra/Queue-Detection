import torch
from PIL import Image
# from detector import Detector_Class
from detector import Detector 
from pprint import pprint

def main():
    detectorModule = Detector()
    model = detectorModule.load_model("yolov5s", pretrained = True)

    path = "../data/images/"
    images = ["bus.jpg", "zidane.jpg"]

    # for image_name in images:
    #     filename = path + image_name
    #     predictions = detectorModule.make_predictions(model, filename)

    #     predictions.print()
    #     predictions.save()


    # Predict People
    for image_name in images:
        filename = path + image_name
        people_predictions = detectorModule.predict_people(model, filename)

        people_predictions.print()
        # people_predictions.save()


main()