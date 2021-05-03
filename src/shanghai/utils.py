'''
Implements some useful utility functions for data processing.  
'''

import numpy as np
import math
import cv2

def get_density_map(image, points):
    image_density = np.zeros_like(image, dtype=np.float64)
    height, width = image_density.shape
    if points is None:
        return image_density
    if points.shape[0] == 1:
        x1 = max(0, min(width-1, round(points[0, 0])))
        y1 = max(0, min(height-1, round(points[0, 1])))
        image_density[y1, x1] = 255
        return image_density
    for j in range(points.shape[0]):
        frame_size = 15
        sigma = 4.0
        Height = np.multiply(cv2.getGaussianKernel(frame_size, sigma), (cv2.getGaussianKernel(frame_size, sigma)).T)
        x = min(width-1, max(0, abs(int(math.floor(points[j, 0])))))
        y = min(height-1, max(0, abs(int(math.floor(points[j, 1])))))
        if x >= width or y >= height:
            continue
        x1 = x - frame_size//2 + 0
        y1 = y - frame_size//2 + 0
        x2 = x + frame_size//2 + 1
        y2 = y + frame_size//2 + 1
        dfx1, dfy1, dfx2, dfy2 = 0, 0, 0, 0
        change_Height = False
        if x1 < 0:
            dfx1 = abs(x1) + 0
            x1 = 0
            change_Height = True
        if y1 < 0:
            dfy1 = abs(y1) + 0
            y1 = 0
            change_Height = True
        if x2 > width:
            dfx2 = x2 - width
            x2 = width
            change_Height = True
        if y2 > height:
            dfy2 = y2 - height
            y2 = height
            change_Height = True
        x1h, y1h, x2h, y2h = 1 + dfx1, 1 + dfy1, frame_size - dfx2, frame_size - dfy2
        if change_Height is True:
            Height = np.multiply(cv2.getGaussianKernel(y2h-y1h+1, sigma), (cv2.getGaussianKernel(x2h-x1h+1, sigma)).T)
        image_density[y1:y2, x1:x2] += Height
 
    return image_density

def x_y_generator(images_path, labels_path, batch_size=64):
    break_point = 0
    t = 0
    images_path = np.squeeze(images_path).tolist() if isinstance(images_path, np.ndarray) else images_path
    labels_path = np.squeeze(labels_path).tolist() if isinstance(labels_path, np.ndarray) else labels_path
    data_length = len(labels_path)
    while True:
        if not break_point:
            x = []
            y = []
            inner_iteration = batch_size
        else:
            t = 0
            inner_iteration = batch_size - data_length % batch_size
        for i in range(inner_iteration):
            if t >= data_length:
                break_point = 1
                break
            else:
                break_point = 0
            img = (cv2.imread(images_path[t], 0) - 127.5) / 128
            density_map = np.loadtxt(labels_path[t], delimiter=',')
            std = 4
            quarter_den = np.zeros((np.asarray(density_map.shape).astype(int)//std).tolist())
            for r in range(quarter_den.shape[0]):
                for c in range(quarter_den.shape[1]):
                    quarter_den[r, c] = np.sum(density_map[r*std:(r+1)*std, c*std:(c+1)*std])
            x.append(img.reshape(*img.shape, 1))
            y.append(quarter_den.reshape(*quarter_den.shape, 1))
            t += 1
        if not break_point:
            x, y = np.asarray(x), np.asarray(y)
            yield x, y

def mean_absolute_error(labels, predictions):
    return K.sum(K.abs(labels - predictions)) / 1
 
def mean_square_error(labels, predictions):
    return K.sum(K.square(labels - predictions)) / 1