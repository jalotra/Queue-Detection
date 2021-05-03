'''
Implements a Keras Model for Object Detection based on RCNN 
'''
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Input
 
def Multi_Column_CNN(input_shape=None):
    inputs = Input(shape=input_shape)
    # first column 
    conv_1 = Conv2D(16, (9, 9), padding='same', activation='relu')(inputs)
    conv_1 = MaxPooling2D(2)(conv_1)
    conv_1 = (conv_1)
    conv_1 = Conv2D(32, (7, 7), padding='same', activation='relu')(conv_1)
    conv_1 = MaxPooling2D(2)(conv_1)
    conv_1 = Conv2D(16, (7, 7), padding='same', activation='relu')(conv_1)
    conv_1 = Conv2D(8, (7, 7), padding='same', activation='relu')(conv_1)
 
    # second column 
    conv_2 = Conv2D(20, (7, 7), padding='same', activation='relu')(inputs)
    conv_2 = MaxPooling2D(2)(conv_2)
    conv_2 = (conv_2)
    conv_2 = Conv2D(40, (5, 5), padding='same', activation='relu')(conv_2)
    conv_2 = MaxPooling2D(2)(conv_2)
    conv_2 = Conv2D(20, (5, 5), padding='same', activation='relu')(conv_2)
    conv_2 = Conv2D(10, (5, 5), padding='same', activation='relu')(conv_2)
 
    # third column 
    conv_3 = Conv2D(24, (5, 5), padding='same', activation='relu')(inputs)
    conv_3 = MaxPooling2D(2)(conv_3)
    conv_3 = (conv_3)
    conv_3 = Conv2D(48, (3, 3), padding='same', activation='relu')(conv_3)
    conv_3 = MaxPooling2D(2)(conv_3)
    conv_3 = Conv2D(24, (3, 3), padding='same', activation='relu')(conv_3)
    conv_3 = Conv2D(12, (3, 3), padding='same', activation='relu')(conv_3)
 
    # merge feature map of third column in last dimension and get density map
    conv_merge = Concatenate(axis=-1)([conv_1, conv_2, conv_3])
    # getting density map as output
    density_map = Conv2D(1, (1, 1), padding='same')(conv_merge)
 
    model = Model(inputs=inputs, outputs=density_map)
    return model