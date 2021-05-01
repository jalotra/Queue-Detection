import torch
from PIL import Image
# from detector import Detector_Class
from detector import Detector 
from pprint import pprint
from convex_hull import ConvexHull


# def main():
#     points = [[1, 2], [3, 4], [2, 3]]
#     hull = ConvexHull(points)
#     print(hull.find_convex_hull())

def main():
    detectorModule = Detector()
    model = detectorModule.load_model("yolov5s", pretrained = True)

    path = "../data/ShanghaiData/"
    images = ["IMG_10.jpg"]

    # Predict People
    for image_name in images:
        filename = path + image_name
        people_predictions = detectorModule.predict_people(model, filename)
        q = detectorModule.predict_queue(people_predictions)
        print(q)

main()