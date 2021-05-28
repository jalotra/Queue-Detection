from typing import List
import torch
from PIL import Image
from math import sqrt

from convex_hull import ConvexHull, Point

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

    def find_distance(self, hull_points : List, m : float, c : float) -> float :
        # assert(isinstance(hull_points[0], Point))
        dist_sum = 0
        for p in hull_points:
            dist_sum += abs(p[1] - m * p[0] - c) / (sqrt(1 + m * m))
        
        return dist_sum
    
    def predict_queue(self, detections_object, show = False):
        '''
        @param [detections_object] = Object of the class models.common.Detections.
        '''
        if detections_object is None:
            raise Exception(f"Points can't be found. Points are {detections_object} ")
        
        points = []
        for *box, _ , _ in detections_object.pred:

            for x in box:
                x = x.cpu().detach().numpy()
                p1, p2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                # Appending the mid point of rectangle to points
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2 
                if show:
                    print(f"Mid_X : {mid_x}, Mid_Y : {mid_y}")
                points.append([mid_x, mid_y])

        print(points)
        # print(f"Convex Hull : {convex_hull}")
        if(len(points) >= 2):
            # Finding the convex hull of these points
            convex_hull = ConvexHull(points).find_convex_hull()
            for idx1 in range(len(convex_hull)):
                for idx2 in range(idx1 + 1, len(convex_hull)):
                    # Slope : Some Big value to start with
                    m, c = 1e+9, 0
                    if(convex_hull[idx2].x - convex_hull[idx1].x != 0):
                        m = (convex_hull[idx2].y - convex_hull[idx1]. y) / (convex_hull[idx2].x - convex_hull[idx1].x)
                    c = -convex_hull[idx1].x * m + convex_hull[idx1].y

                    best_m, best_c, best_dist = -1, -1, 1e+18 
                    if(self.find_distance(points, m, c) < best_dist):
                        best_dist = self.find_distance(points, m, c)
                        best_m = m
                        best_c = c

            # Now find the best points possible
            # Using the slope and intercept
            thresold = 0.90
            max_dist = 0
            for p in points:
                dist_here = abs(p[1] - best_m * p[0] - best_c) / (sqrt(1 + best_m * best_m))
                max_dist = max(max_dist, dist_here)
            print(f"Max Distance : {max_dist}")
            resultant_points = []
            for p in points:
                dist_here = abs(p[1] - best_m * p[0] - best_c) / (sqrt(1 + best_m * best_m))
                print(f"Distance Here : {dist_here}")
                if(dist_here <= thresold * max_dist):
                    resultant_points.append([p[0], p[1]])
            return resultant_points
            print(f"Resultant Points : {resultant_points}")
        else:
            print(f"Resultant Points : {points}")
            return points
        return resultant_points
        



