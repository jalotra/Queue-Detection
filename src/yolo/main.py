import torch
from PIL import Image
# from detector import Detector_Class
from detector import Detector 
from pprint import pprint
from convex_hull import ConvexHull
import glob
import cv2


# def main():
#     points = [[1, 2], [3, 4], [2, 3]]
#     hull = ConvexHull(points)
#     print(hull.find_convex_hull())

def main():
    detectorModule = Detector()
    model = detectorModule.load_model("yolov5x", pretrained = True)

    images = glob.glob("../data/images/*.jpg")
    # print(images[0])

    # Predict People
    for image_name in images:
        filename = image_name
        print(f"FILENAME : {filename}")
        people_predictions = detectorModule.predict_people(model, filename)
        # people_predictions.show()
        q = detectorModule.predict_queue(people_predictions)
        
        image = cv2.imread(image_name)
        for i, p in enumerate(q):
            image = cv2.circle(image, (p[0],p[1]), radius=10, color=(0, 0, 255), thickness=-1)
        
        cv2.imwrite("./results/" + image_name.split('/')[-1].split('.')[0] + "_result.jpg", image)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #   continue

main()