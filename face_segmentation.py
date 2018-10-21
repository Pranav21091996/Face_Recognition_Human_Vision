from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
import numpy as np 
import os
import matplotlib.pyplot as plt 
import cv2
import scipy.ndimage as ndimage
import time

from FCN8s_keras import FCN

model = FCN()
model.load_weights("Keras_FCN8s_face_seg_YuvalNirkin.h5")

def vgg_preprocess(im):
    im = cv2.resize(im, (500, 500))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_[np.newaxis,:]
    return in_
  
def auto_downscaling(im):
    w = im.shape[1]
    h = im.shape[0]
    while w*h >= 700*700:
        im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
        w = im.shape[1]
        h = im.shape[0]
    return im

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

facedir = 'Cropped_Face'
FaceMask = "FaceMask"
BinaryMask = "BinaryMask" 
if not os.path.exists(BinaryMask):
    os.makedirs(BinaryMask)

if not os.path.exists(FaceMask):
    os.makedirs(FaceMask)

for directory in get_immediate_subdirectories(facedir):
    
    if not os.path.exists(os.path.join(FaceMask, directory)):
        os.makedirs(os.path.join(FaceMask, directory))
    if not os.path.exists(os.path.join(BinaryMask, directory)):
        os.makedirs(os.path.join(BinaryMask, directory))
 
    for filename in os.listdir(os.path.join(facedir, directory)):
 
        basename =  os.path.splitext(filename)[0]
        try:
            im = cv2.cvtColor(cv2.imread(os.path.join(facedir, directory,filename)), cv2.COLOR_BGR2RGB)
            im = auto_downscaling(im)
            inp_im = vgg_preprocess(im)
            out = model.predict([inp_im])
            out_resized = cv2.resize(np.squeeze(out), (im.shape[1],im.shape[0]))
            out_resized_clipped = np.clip(out_resized.argmax(axis=2), 0, 1).astype(np.float64)
            mask = cv2.GaussianBlur(out_resized_clipped, (7,7), 6)
            plt.imsave(os.path.join(FaceMask, directory,filename),(mask[:,:,np.newaxis]*im.astype(np.float64)).astype(np.uint8))
            plt.imsave(os.path.join(BinaryMask, directory,filename),out_resized_clipped, cmap='gray')
        except Exception as e:	
            print(e)

