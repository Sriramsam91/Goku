
# coding: utf-8

# **Outline of Steps**
#     + Initialization
#         + Download COCO detection data from http://cocodataset.org/#download
#             + http://images.cocodataset.org/zips/train2014.zip <= train images
#             + http://images.cocodataset.org/zips/val2014.zip <= validation images
#             + http://images.cocodataset.org/annotations/annotations_trainval2014.zip <= train and validation annotations
#         + Run this script to convert annotations in COCO format to VOC format
#             + https://gist.github.com/chicham/6ed3842d0d2014987186#file-coco2pascal-py
#         + Download pre-trained weights from https://pjreddie.com/darknet/yolo/
#             + https://pjreddie.com/media/files/yolo.weights
#         + Specify the directory of train annotations (train_annot_folder) and train images (train_image_folder)
#         + Specify the directory of validation annotations (valid_annot_folder) and validation images (valid_image_folder)
#         + Specity the path of pre-trained weights by setting variable *wt_path*
#     + Construct equivalent network in Keras
#         + Network arch from https://github.com/pjreddie/darknet/blob/master/cfg/yolo-voc.cfg
#     + Load the pretrained weights
#     + Perform training 
#     + Perform detection on an image with newly trained weights
#     + Perform detection on an video with newly trained weights

# # Initialization

# In[1]:


from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, cv2
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes, normalize
from keras.models import load_model



# In[2]:


#LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
LABELS = ['gooddayOrangeCashew', 'parleg']
#LABELS = ['RBC']
IMAGE_H, IMAGE_W = 256, 256
GRID_H  = [8, 16, 32]  
GRID_W  =  [8, 16, 32]
BOX              = 3
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3#0.5
NMS_THRESHOLD    = 0.3#0.45
#ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS          = [0.66, 0.35, 1.13, 1.21, 1.96, 0.73]

NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 4
WARM_UP_BATCHES  = 2
TRUE_BOX_BUFFER  = 50


# In[3]:


wt_path = '/home/sathish/Downloads/darknet/yolov3.weights'                       
train_image_folder = '../../gitRepos/YOLO/darkflow/custom_dataset/BT/Train/Img/'
train_annot_folder = '../../gitRepos/YOLO/darkflow/custom_dataset/BT//Train/Annotation/'
valid_image_folder = '../../gitRepos/YOLO/darkflow/custom_dataset/BT/Validation/Img/'
valid_annot_folder = '../../gitRepos/YOLO/darkflow/custom_dataset/BT/Validation/Annotation/'
import os
print(len(os.listdir(valid_image_folder)))


# 
# # Construct the network

# In[4]:


# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)
    
def space_to_depth_x3(x):
    return tf.space_to_depth(x, block_size=4)
    
def space_to_depth_x4(x):
    return tf.space_to_depth(x, block_size=8)
    
def space_to_depth_x5(x):
    return tf.space_to_depth(x, block_size=16)
def space_to_depth_x6(x):
    return tf.space_to_depth(x, block_size=6)    



# In[5]:


input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, Activation, LeakyReLU, Conv2DTranspose, Lambda, GlobalAveragePooling2D
import keras.backend as K
input_shape = (256, 256, 3)

input_z = Input(shape=input_shape)

x = Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same')(input_z)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
#It's been shown that instead of using a maxpooling, a conv2d with a strides 2 does better. Dilated CNN also work
x = Conv2D(64, (3, 3), strides = (2, 2), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
print('after 1st stride', K.int_shape(x))
#512*512*64
#Residual Layers
x = Conv2D(32, (1, 1),kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c1 = x = LeakyReLU(alpha=0.3)(x)
c1 = Lambda(space_to_depth_x5)(c1)
print('After first residual network- c1', K.int_shape(c1))
#512*512*64
#Downsampling Layer
x = Conv2D(128, (3, 3), strides = (2, 2), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
print('After second striding', K.int_shape(x))
#256*256*128
#---****
x = Conv2D(64, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c2 = x = LeakyReLU(alpha=0.3)(x)
c2 = Lambda(space_to_depth_x4)(c2)
#skip_connection = Lambda(space_to_depth_x2)(c1)
#c2 = concatenate([c2, skip_connection])
print('c2', K.int_shape(c2))
x = Conv2D(64, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c3 = x = LeakyReLU(alpha=0.3)(x)
c3 = Lambda(space_to_depth_x4)(c3)
#print('skip_connection to c3', K.int_shape(skip_connection))
#c3 = concatenate([skip_connection, c3])
print('c3 shape', K.int_shape(c3))
#---*******
#256*256*128
x = Conv2D(256, (3, 3), strides = (2, 2), kernel_initializer='he_uniform', activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
#------*****
#128*128*256
x = Conv2D(128, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer= 'he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c4 = x = LeakyReLU(alpha=0.3)(x)
c4 = Lambda(space_to_depth_x3)(c4)
#skip_connection = Lambda(space_to_depth_x2)(c3)
#c4 = concatenate([c4, skip_connection])
x = Conv2D(128, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer= 'he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c5 = x = LeakyReLU(alpha=0.3)(x)
c5 = Lambda(space_to_depth_x3)(c5)
#skip_connection = Lambda(space_to_depth_x2)(c4)
#c5 = concatenate([c5, skip_connection])
x = Conv2D(128, (1, 1), kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer= 'he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c6 = x = LeakyReLU(alpha=0.3)(x)
c6 = Lambda(space_to_depth_x3)(c6)
#skip_connection = Lambda(space_to_depth_x2)(c5)
#c6 = concatenate([skip_connection, c6])
x = Conv2D(128, (1, 1), kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer= 'he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c7 = x = LeakyReLU(alpha=0.3)(x)
c7 = Lambda(space_to_depth_x3)(c7)
#skip_connection = Lambda(space_to_depth_x2)(c6)
#c7 = concatenate([skip_connection, c7])
x = Conv2D(128, (1, 1), kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer= 'he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c8 = x = LeakyReLU(alpha=0.3)(x)
c8 = Lambda(space_to_depth_x3)(c8)
#skip_connection = Lambda(space_to_depth_x2)(c7)
#c8 = concatenate([skip_connection, c8])
x = Conv2D(128, (1, 1), kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer= 'he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c9 = x = LeakyReLU(alpha=0.3)(x)
c9 = Lambda(space_to_depth_x3)(c9)
#skip_connection = Lambda(space_to_depth_x2)(c8)
#c9 = concatenate([skip_connection, c9])
x = Conv2D(128, (1, 1), kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer= 'he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c10 = x = LeakyReLU(alpha=0.3)(x)
c10 = Lambda(space_to_depth_x3)(c10)
#skip_connection = Lambda(space_to_depth_x2)(c9)
#c10 = concatenate([skip_connection, c10])
x = Conv2D(128, (1, 1), kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer= 'he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c11_1 = x = LeakyReLU(alpha=0.3)(x)
c11 = Lambda(space_to_depth_x3)(c11_1)
#skip_connection = Lambda(space_to_depth_x2)(c3)
#c11 = concatenate([skip_connection, c11])
print('c11 shape', K.int_shape(c11))
#----*****
#128*128*256
x = Conv2D(512, (3, 3), strides=(2, 2), kernel_initializer='he_uniform', activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
print('After c11 striding', K.int_shape(x))
#64*64*512
#-------************
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c12 = x = LeakyReLU(alpha=0.3)(x)
c12 = Lambda(space_to_depth_x2)(c12)
#skip_connection = Lambda(space_to_depth_x2)(c11)
#c12 = concatenate([c12, skip_connection])
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c13 = x = LeakyReLU(alpha=0.3)(x)
c13 = Lambda(space_to_depth_x2)(c13)
#skip_connection = Lambda(space_to_depth_x2)(c13)
#c13 = concatenate([skip_connection, c12])
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c14 = x = LeakyReLU(alpha=0.3)(x)
c14 = Lambda(space_to_depth_x2)(c14)
#skip_connection = Lambda(space_to_depth_x2)
#c14 = concatenate([skip_connection, c14])
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c15 = x = LeakyReLU(alpha=0.3)(x)
c15 = Lambda(space_to_depth_x2)(c15)
#skip_connection = Lambda(space_to_depth_x2)(c14)
#c15 = concatenate([skip_connection, c15])
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c16 = x = LeakyReLU(alpha=0.3)(x)
c16 = Lambda(space_to_depth_x2)(c16)
#skip_connection = Lambda(space_to_depth_x2)(c15)
#c16 = concatenate([skip_connection, c16])
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c17 = x = LeakyReLU(alpha=0.3)(x)
c17 = Lambda(space_to_depth_x2)(c17)
#skip_connection = Lambda(space_to_depth_x2)(c16)
#c17 = concatenate([skip_connection, c17])
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c18 = x = LeakyReLU(alpha=0.3)(x)
c18 = Lambda(space_to_depth_x2)(c18)
#skip_connection = Lambda(space_to_depth_x2)(c17)
#c18 = concatenate([skip_connection, c18])
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c19_1 = x = LeakyReLU(alpha=0.3)(x)
c19 = Lambda(space_to_depth_x2)(c19_1)
#c19 = concatenate([skip_connection, c19])
#print('c19 shape',K.int_shape(c19))
#-----*****************
#64*64*512

x = Conv2D(1024, (3, 3), strides=(2, 2), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
print('After c19 striding', K.int_shape(x))
#32*32*1024
#------************
x = Conv2D(512, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x= Conv2D(1024, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c20 = x = LeakyReLU(alpha=0.3)(x)
#c20 = Lambda(space_to_depth_x2)(c20)
#skip_connection = Lambda(space_to_depth_x2)(c19)
#c20 = concatenate([c19, c20])
x = Conv2D(512, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x= Conv2D(1024, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c21 = x = LeakyReLU(alpha=0.3)(x)
#c21 = Lambda(space_to_depth_x2)(c21)
#skip_connection = Lambda(space_to_depth_x2)(c20)
#c21 = concatenate([skip_connection, c21])
x = Conv2D(512, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x= Conv2D(1024, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c22 = x = LeakyReLU(alpha=0.3)(x)
#c22 = Lambda(space_to_depth_x2)(c22)
#skip_connection = Lambda(space_to_depth_x2)(c21)
#c22 = concatenate([skip_connection, c22])
x = Conv2D(512, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x= Conv2D(1024, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c23 = x = LeakyReLU(alpha=0.3)(x)
#c23 = Lambda(space_to_depth_x2)(c23)
#c23 = concatenate([skip_connection, c23])
print('shape at c23', K.int_shape(c23))
#32*32*1024
#-----******************
x = Conv2D(512, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
#32*32*512
x= Conv2D(1024, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
#32*32*1024
x = Conv2D(512, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
#32*32*512
x = Conv2D(1024, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c24 =x = LeakyReLU(alpha=0.3)(x)
#skip_connection = Lambda(space_to_depth_x2)(c23)
#c24 = concatenate([c23, c24])
print('c24 shape', K.int_shape(c24))
#32*32*1024
x = Conv2D(512, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
c25 = x = LeakyReLU(alpha=0.3)(x)
print('shape after final residual', K.int_shape(x))
#skip_connection = x = concatenate([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, x], axis = 3)
skip_connection = x = concatenate([c3, c9, c19, c23, x], axis = 3)
x = Activation('linear')(x)
#x = concatenate([c24, x])
#32*32*512

x = Conv2D(1024, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
#32*32*512
D1 = Conv2D(21, (1, 1), kernel_initializer='he_uniform', activation='linear')(x)
print(K.int_shape(D1))
#32*32*255
#c24 - 32*32*1024
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same', name='d2_conv')(c25)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
print('shape before deconv', K.int_shape(x))
#32*32*256
print('Before upsampling', K.int_shape(x))
c98 = x = Conv2DTranspose(512, (1, 1), strides=(2, 2), kernel_initializer='he_uniform', activation='relu')(x)
#print('After upsampling', K.eval(c98))

#64*64*512
x = concatenate([x, c19_1])
x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)

x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)

x = Conv2D(256, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(512, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
d2 = Conv2D(21, (1, 1), kernel_initializer='he_uniform', activation='linear', padding='same')(x)
#Call Yolo Anchors, Mask, Loss
#64*64*512
x = Conv2D(128, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
#64*64*128
x = Conv2DTranspose(256, (1, 1), strides=(2, 2), kernel_initializer='he_uniform', activation='relu', padding='same')(x)
#128*128*256
#c11 128*128*256
x = concatenate([x, c11_1])

x = Conv2D(128, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x= BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)

x = Conv2D(128, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x= BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)

x = Conv2D(128, (1, 1), kernel_initializer='he_uniform', padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(256, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
x= BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)

d3 = x = Conv2D(21, (1, 1), kernel_initializer='he_uniform', padding='same', activation='linear')(x)

#x= GlobalAveragePooling2D()([D1, d2, d3])
#model = Model(input_z, [D1, d2, d3])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()

# Layer 23
#x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
output1 = Reshape((GRID_H[0], GRID_W[0], BOX, 4 + 1 + CLASS))(D1)
output2 = Reshape((GRID_H[1], GRID_W[1], BOX, 4 + 1+ CLASS))(d2)
output3 = Reshape((GRID_H[2], GRID_W[2], BOX, 4 + 1 + CLASS))(d3)


# small hack to allow true_boxes to be registered when Keras build the model 
# for more information: https://github.com/fchollet/keras/issues/2790
output1 = Lambda(lambda args: args[0])([output1, true_boxes])
output2 = Lambda(lambda args: args[0])([output2, true_boxes])
output3 = Lambda(lambda args: args[0])([output3, true_boxes])
output = [output2, output1, output3]

model = Model(inputs=[input_z, true_boxes], outputs = [output1, output2, output3])


# In[6]:
#print('output_shape', (model.output))

model.summary()
print('FINISHED SUMMARY')

#model.save('mcgroce1.h5')


# # Load pretrained weights

# **Load the weights originally provided by YOLO**

# In[7]:


weight_reader = WeightReader(wt_path)


# In[8]:


weight_reader.reset()
nb_conv = 53

for i in range(1, nb_conv+1):
    conv_layer = model.get_layer('conv2d_' + str(i))
    
    if i < nb_conv:
        norm_layer = model.get_layer('batch_normalization_' + str(i))
        
        size = np.prod(norm_layer.get_weights()[0].shape)

        beta  = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean  = weight_reader.read_bytes(size)
        var   = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])       
        
    if len(conv_layer.get_weights()) > 1:
        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])
print('finished loading weights')

# **Randomize weights of the last layer**

# In[9]:


#layer   = model.layers[-4] # the last convolutional layer
#weights = layer.get_weights()

#new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
#new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

#layer.set_weights([new_kernel, new_bias])
#print(layer)


# # Perform training

# **Loss function**

# $$\begin{multline}
# \lambda_\textbf{coord}
# \sum_{i = 0}^{S^2}
#     \sum_{j = 0}^{B}
#      L_{ij}^{\text{obj}}
#             \left[
#             \left(
#                 x_i - \hat{x}_i
#             \right)^2 +
#             \left(
#                 y_i - \hat{y}_i
#             \right)^2
#             \right]
# \\
# + \lambda_\textbf{coord} 
# \sum_{i = 0}^{S^2}
#     \sum_{j = 0}^{B}
#          L_{ij}^{\text{obj}}
#          \left[
#         \left(
#             \sqrt{w_i} - \sqrt{\hat{w}_i}
#         \right)^2 +
#         \left(
#             \sqrt{h_i} - \sqrt{\hat{h}_i}
#         \right)^2
#         \right]
# \\
# + \sum_{i = 0}^{S^2}
#     \sum_{j = 0}^{B}
#         L_{ij}^{\text{obj}}
#         \left(
#             C_i - \hat{C}_i
#         \right)^2
# \\
# + \lambda_\textrm{noobj}
# \sum_{i = 0}^{S^2}
#     \sum_{j = 0}^{B}
#     L_{ij}^{\text{noobj}}
#         \left(
#             C_i - \hat{C}_i
#         \right)^2
# \\
# + \sum_{i = 0}^{S^2}
# L_i^{\text{obj}}
#     \sum_{c \in \textrm{classes}}
#         \left(
#             p_i(c) - \hat{p}_i(c)
#         \right)^2
# \end{multline}$$

# In[7]:


def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]
    print('mask_shape', K.int_shape(mask_shape))
    GRID_W = [8, 16, 32]
    GRID_H = [8, 16, 32]
    total_loss = []
    nb_coord_box_loss = []
    nb_conf_box_loss = []
    nb_class_box_loss = []
    loss_xy_loss = []
    loss_wh_loss = []
    loss_conf_loss = []
    loss_class_loss = []
    z =  K.int_shape(y_pred)
    print(z[1])
    print('shape of y_true', K.int_shape(y_true))
    #for i in range(0, 3):
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(z[1]), [z[1]]), (1, z[1], z[1], 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))

    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 3, 1])
    
    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)
    
    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)
    
    """
    Adjust prediction
    """
    ### adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    print(K.int_shape(pred_box_xy))
    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    
    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    
    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]
    
    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell
    
    ### adjust w and h
    true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
    
    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half
    
    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half       
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    #print(cell_x)
    true_box_conf = iou_scores * y_true[..., 4]
    
    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    
    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    
    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
    
    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
    
    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)
    seen = tf.assign_add(seen, 1.)
    
    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask, 
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy, 
                                   true_box_wh,
                                   coord_mask])
    
    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    
    loss = loss_xy + loss_wh + loss_conf + loss_class
    
    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
    #nb_coord_box_loss.append(nb_coord_box)
    #nb_conf_box_loss.append(nb_conf_box)
    #nb_class_box_loss.append(nb_class_box)
    #loss_xy_loss.append(loss_xy)
    #loss_wh_loss.append(loss_wh)
    #loss_conf_loss.append(loss_conf_loss)
    #loss_class_loss.append(loss_class)
    #total_loss.append(loss)
    """
    Debugging code
    """    
    current_recall = nb_pred_box/(nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall) 

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
  
    return loss


# **Parse the annotations to construct train generator and validation generator**

# In[8]:


generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 50,
}
print(generator_config['LABELS'])
import xml.etree.ElementTree as ET
from utils import BoundBox, normalize, bbox_iou
def parse_annotation1(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
        
        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
                
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
        
               
    return all_imgs, seen_labels
            


# In[9]:


train_imgs, seen_train_labels = parse_annotation1(train_annot_folder, train_image_folder, labels=LABELS)
### write parsed annotations to pickle for fast retrieval next time
#with open('train_imgs', 'wb') as fp:
#    pickle.dump(train_imgs, fp)

### read saved pickle of parsed annotations
#with open ('train_imgs', 'rb') as fp:
#    train_imgs = pickle.load(fp)
train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize)
#print(K.int_shape(train_batch))
#plt.imshow(train_imgs[0])
#lt.show()
print(len(train_imgs))
print(seen_train_labels)
valid_imgs, seen_valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels=LABELS)
### write parsed annotations to pickle for fast retrieval next time
#with open('valid_imgs', 'wb') as fp:
#    pickle.dump(valid_imgs, fp)



### read saved pickle of parsed annotations
#with open ('valid_imgs', 'rb') as fp:
#    valid_imgs = pickle.load(fp)
valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)


# **Setup a few callbacks and start the training**

# In[10]:


#model = load_model('weights_mcgroce_yolov3_1.h5')
early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=55, 
                           mode='min', 
                           verbose=1)
#from keras.models import load_model
#import tensorflow as tf
#model = load_model('mcgroce_2.h5')

checkpoint = ModelCheckpoint('weights_mcgroce_yolov3_1.h5', 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)
from keras.callbacks import LearningRateScheduler
#lr_schedule = lambda epoch: 0.001 if epoch < 40 else 0.0001
#callback = [LearningRateScheduler(lr_schedule)]


# In[ ]:


#tb_counter  = len([log for log in os.listdir(os.path.expanduser('~/logs/')) if 'coco_' in log]) + 1
#tensorboard = TensorBoard(log_dir = './logs', histogram_freq=0, 
#                          write_graph=True, #
#                          write_images=False)

optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss={'lambda_20': custom_loss, 'lambda_21': custom_loss, 'lambda_22': custom_loss}, optimizer=optimizer)
#with tf.device('/gpu:0'):
history = model.fit_generator(generator        = train_batch, 
                              steps_per_epoch  = len(train_batch), 
                              epochs           = 1000, 
                              verbose          = 1,
                              validation_data  = valid_batch,
                              validation_steps = len(valid_batch),
                              callbacks        = [checkpoint], 
                              max_queue_size   = 1)


# In[17]:

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#fig = matplotlib.pyplot.gcf()
#plt.set_size_inches(18.5, 10.5)

plt.savefig('model_train_val1.jpg')


# # Perform detection on image

# In[20]:


model.save_weights("mcgroce_2.h5")
model.save('mcgroce_1.h5')
dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))


# In[21]:


image = cv2.imread('../../gitRepos/YOLO/darkflow/custom_dataset/BT/Train/Img/MBBM_10_Grocery_Parle-G.jpg')
plt.figure(figsize=(10,10))
#model_load('mcgroce_1.h5')
input_image = cv2.resize(image, (416, 416))
input_image = input_image / 255.
input_image = input_image[:,:,::-1]
input_image = np.expand_dims(input_image, 0)

netout = model.predict([input_image, dummy_array])

boxes = decode_netout(netout[0], 
                      obj_threshold=OBJ_THRESHOLD,
                      nms_threshold=NMS_THRESHOLD,
                      anchors=ANCHORS, 
                      nb_class=CLASS)
image = draw_boxes(image, boxes, labels=LABELS)

plt.imshow(image[:,:,::-1]); plt.show()
#plt.savefig('mcgroce241.jpg')
#from keras import backend as k
#for box in boxes:
#    print(box.get_score())


# # Perform detection on video

# In[ ]:


#model.load_weights("weights_coco.h5")

#dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))


# In[ ]:


#video_inp = '../basic-yolo-keras/images/phnom_penh.mp4'
#video_out = '../basic-yolo-keras/images/phnom_penh_bbox.mp4'

#video_reader = cv2.VideoCapture(video_inp)

#nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
#frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
#frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

#video_writer = cv2.VideoWriter(video_out,
#                               cv2.VideoWriter_fourcc(*'XVID'), 
#                               50.0, 
#                               (frame_w, frame_h))

#for i in tqdm(range(nb_frames)):
#    ret, image = video_reader.read()
    
#    input_image = cv2.resize(image, (416, 416))
#    input_image = input_image / 255.
#    input_image = input_image[:,:,::-1]
#    input_image = np.expand_dims(input_image, 0)

#    netout = model.predict([input_image, dummy_array])

#    boxes = decode_netout(netout[0], 
#                          obj_threshold=0.3,
#                          nms_threshold=NMS_THRESHOLD,
#                          anchors=ANCHORS, 
#                          nb_class=CLASS)
#    image = draw_boxes(image, boxes, labels=LABELS)

#    video_writer.write(np.uint8(image))
    
#video_reader.release()
#video_writer.release()  

