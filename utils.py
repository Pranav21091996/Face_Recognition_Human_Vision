import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import sys

def normalize(x):

    a = 0.
    b = 1.
    min = 0
    max = 255

    return a + (((x - min)*(b - a))/(max - min))

def crop_center_resize(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    img = img[starty:starty+cropy,startx:startx+cropx,:]
    return cv2.resize(img, (64,64))

def one_hot_encode(x,num_classes):
	encoder = preprocessing.LabelBinarizer()
	list1 = np.arange(0,num_classes)
	encoder.fit(list1)
	x = encoder.transform(x)
	return x

def neural_net_image_input():

	x = tf.placeholder(tf.float32, shape = [None, 4,64,64,3], name = 'x')
	return x

def neural_net_label_input(n_classes):

	y = tf.placeholder(tf.float32, shape = [None, n_classes], name = 'y')
	return y

def neural_net_keep_prob_input():

	keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
	return keep_prob

def conv3d(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides,in_channels):

	if(x_tensor.get_shape()[2] < 5 ):
		F_W = tf.Variable(tf.truncated_normal([1,1,1, in_channels,  conv_num_outputs], stddev=0.05, mean=0.0))
	else:
		F_W = tf.Variable(tf.truncated_normal([1,conv_ksize[0], conv_ksize[1], in_channels,  conv_num_outputs], stddev=0.05, mean=0.0))
	F_b = tf.Variable(tf.zeros(conv_num_outputs))

	layer1 = tf.nn.conv3d(x_tensor,
						  F_W,
						  strides=[1, 1, 1, 1, 1],
						  padding = 'VALID')
	layer2a = tf.nn.bias_add(layer1, F_b)
	layer2b = tf.nn.relu(layer2a)
	return layer2b

def maxpool3d(conv,pool_ksize,pool_strides,flag):
	if(flag == 0):
		layer2c = tf.nn.max_pool3d(
				conv,
				ksize=[1, 1, pool_ksize[0], pool_ksize[1], 1],
				strides=[1, 1, pool_strides[0], pool_strides[1], 1],
				padding = 'VALID')
	elif(flag == 1):
		kernel_size = conv.get_shape()[2]
		layer2c = tf.nn.max_pool3d(
				conv,
				ksize=[1, 1, kernel_size, kernel_size, 1],
				strides=[1, 1, pool_strides[0], pool_strides[1], 1],
				padding = 'VALID')
	else:
		layer2c = tf.nn.max_pool3d(
				conv,
				ksize=[1, 4, 1, 1, 1],
				strides=[1, 1, 1, 1, 1],
				padding = 'VALID')

	return layer2c

def flatten(x_tensor):

	shape = x_tensor.get_shape().as_list()
	dim = np.prod(shape[1:])
	x_tensor_flat = tf.reshape(x_tensor, [-1, dim])

	return x_tensor_flat

def fully_conn(x_tensor, num_outputs):

	inputs = x_tensor.get_shape().as_list()[1]
	weights = tf.Variable(tf.truncated_normal([inputs, num_outputs], stddev = 0.05, mean = 0.0))
	bias = tf.Variable(tf.zeros(num_outputs))
	logits = tf.add(tf.matmul(x_tensor,weights), bias)

	return tf.nn.relu(logits)

def output(x_tensor, num_outputs):

	inputs = x_tensor.get_shape().as_list()[1]
	weights = tf.Variable(tf.truncated_normal([inputs, num_outputs], stddev = 0.05, mean = 0.0))
	bias = tf.Variable(tf.zeros(num_outputs))
	logits = tf.add(tf.matmul(x_tensor,weights), bias)

	return logits

def conv_net(x, keep_prob):

	x_tensor = x
	conv_ksize = (5,5)
	conv_strides = (1,1)
	pool_ksize = (3,3)
	pool_strides = (1,1)

	flag = 0
	conv_num_outputs = 192
	in_channels = 3
	conv = conv3d(x_tensor,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv = maxpool3d(conv,pool_ksize,pool_strides,flag)

	conv_num_outputs = 192
	in_channels = 192
	conv2 = conv3d(conv,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv2 = maxpool3d(conv2,pool_ksize,pool_strides,flag)

	conv_num_outputs = 192
	in_channels = 192
	conv3 = conv3d(conv2,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv3 = maxpool3d(conv3,pool_ksize,pool_strides,flag)

	conv_num_outputs = 192
	in_channels = 192
	conv4 = conv3d(conv3,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv4 = maxpool3d(conv4,pool_ksize,pool_strides,flag)

	conv_num_outputs = 256
	in_channels = 192
	flag = 1
	conv5 = conv3d(conv4,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv5 = maxpool3d(conv5,pool_ksize,pool_strides,flag)
	flag = 2
	conv5 = maxpool3d(conv5,pool_ksize,pool_strides,flag)


	flat = flatten(conv5)

	fc1 = fully_conn(flat,512)
	fc1 = tf.nn.dropout(fc1,keep_prob)

	out = output(fc1, 100)

	return out

def weight_variable(shape):
    initial = tf.ones(shape,tf.float32)
    #initial = tf.truncated_normal(shape,stddev=0.1)
    return initial
    #return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return initial
    #return tf.Variable(initial)

def dilated_conv_net(x, keep_prob):
	crop32 = tf.image.resize_images(tf.reshape(tf.image.central_crop(x, scale[0]),[-1,32,32,3]),(64,64))
	crop64 = tf.image.resize_images(tf.reshape(tf.image.central_crop(x, scale[1]),[-1,64,64,3]),(64,64))
	crop128 = tf.image.resize_images(tf.reshape(tf.image.central_crop(x, scale[2]),[-1,128,128,3]),(64,64))
	crop256 = tf.image.resize_images(tf.reshape(tf.image.central_crop(x, scale[3]),[-1,256,256,3]),(64,64))
	
	W_conv1 = weight_variable([3,3,3,3])
	b_conv1 = bias_variable([3])
	c_dilat1 =tf.image.resize_images(tf.nn.relu(tf.nn.conv2d(crop32, W_conv1, strides=[1,1,1,1],padding='SAME',
	   dilations=[1, 1, 1, 1], name='dilation1')+b_conv1),(64,64))
	
	W_conv2 = weight_variable([3,3,3,3])
	b_conv2 = bias_variable([3])
	c_dilat2 =tf.image.resize_images(tf.nn.relu(tf.nn.conv2d(crop64, W_conv2, strides=[1,1,1,1],padding='SAME',
	   dilations=[1, 2, 2, 1], name='dilation2')+b_conv2),(64,64))
	
	W_conv3 = weight_variable([3,3,3,3])
	b_conv3 = bias_variable([3])
	c_dilat3 =tf.image.resize_images(tf.nn.relu(tf.nn.conv2d(crop128, W_conv3, strides=[1,1,1,1],padding='SAME',
	   dilations=[1, 4, 4, 1], name='dilation3')+b_conv3),(64,64))
	W_conv4 = weight_variable([3,3,3,3])
	b_conv4 = bias_variable([3])
	c_dilat4 =tf.image.resize_images(tf.nn.relu(tf.nn.conv2d(crop256, W_conv4, strides=[1,1,1,1],padding='SAME',
	   dilations=[1, 8, 8, 1], name='dilation4')+b_conv4),(64,64))
	
	input_x = tf.stack([c_dilat1,c_dilat2,c_dilat3,c_dilat4],axis=1)
	
	x_tensor = input_x
	conv_ksize = (5,5)
	conv_strides = (1,1)
	pool_ksize = (3,3)
	pool_strides = (1,1)

	flag = 0
	conv_num_outputs = 100
	in_channels = 3
	conv = conv3d(x_tensor,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv = maxpool3d(conv,pool_ksize,pool_strides,flag)

	conv_num_outputs = 100
	in_channels = 100
	conv2 = conv3d(conv,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv2 = maxpool3d(conv2,pool_ksize,pool_strides,flag)

	conv_num_outputs = 100
	in_channels = 100
	conv3 = conv3d(conv2,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv3 = maxpool3d(conv3,pool_ksize,pool_strides,flag)

	conv_num_outputs = 100
	in_channels = 100
	conv4 = conv3d(conv3,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv4 = maxpool3d(conv4,pool_ksize,pool_strides,flag)

	conv_num_outputs = 128
	in_channels = 100
	flag = 1
	conv5 = conv3d(conv4,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides,in_channels)
	conv5 = maxpool3d(conv5,pool_ksize,pool_strides,flag)
	flag = 2
	conv5 = maxpool3d(conv5,pool_ksize,pool_strides,flag)


	flat = flatten(conv5)

	fc1 = fully_conn(flat,512)
	fc1 = tf.nn.dropout(fc1,keep_prob)

	logits = output(fc1, 10)

    	return logits


