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
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def dilated_conv_net(x, keep_prob):
  
    #first convolutional layer,kernal size 3*3,dilation rate=1, 3 input channel,32 output channel
    W_conv1 = weight_variable([3,3,3,32])
    b_conv1 = bias_variable([32])

    c_dilat1 =tf.nn.conv2d(x, W_conv1, strides=[1,1,1,1],padding='SAME',
       dilations=[1, 1, 1, 1], name='dilation1')
    h_conv1 = tf.nn.relu(c_dilat1+b_conv1)

    #second convolutional layer
    W_conv2 = weight_variable([3,3,32,32])
    b_conv2 = bias_variable([32])
    c_dilat2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1,1,1,1], padding= 'SAME',
     dilations=[1, 1, 1, 1], name= 'dilation2')
    h_conv2 = tf.nn.relu(c_dilat2+b_conv2)

    #third convolutional layer
    W_conv3 = weight_variable([3,3,32,32])
    b_conv3 = bias_variable([32])
    c_dilat3 = tf.nn.conv2d(h_conv2, W_conv3, strides= [1,1,1,1], padding= 'SAME',
     dilations= [1, 2, 2, 1], name= 'dilation3')
    h_conv3 = tf.nn.relu(c_dilat3 + b_conv3)

    #fourth convolutional layer
    W_conv4 = weight_variable([3,3,32,32])
    b_conv4 = bias_variable([32])
    c_dilat4 = tf.nn.conv2d(h_conv3, W_conv4, strides= [1,1,1,1], padding= 'SAME',
     dilations= [1, 4, 4, 1], name='dilation4')
    h_conv4 = tf.nn.relu(c_dilat4 + b_conv4)
  
  #fifth convolutional layer
    W_conv5 = weight_variable([3,3,32,32])
    b_conv5 = bias_variable([32])
    c_dilat5 = tf.nn.conv2d(h_conv4, W_conv5, strides= [1,1,1,1], padding= 'SAME',
     dilations= [1, 8, 8, 1], name='dilation5')
    h_conv5 = tf.nn.relu(c_dilat5 + b_conv5)
  
  #sixth convolutional layer
    W_conv6 = weight_variable([3,3,32,32])
    b_conv6 = bias_variable([32])
    c_dilat6 = tf.nn.conv2d(h_conv5, W_conv6, strides= [1,1,1,1], padding= 'SAME',
     dilations= [1, 16, 16, 1], name='dilation6')
    h_conv6 = tf.nn.relu(c_dilat6 + b_conv6)

    #seventh convolutional layer
    W_conv7 = weight_variable([3,3,32,32])
    b_conv7 = bias_variable([32])
    c_dilat7 = tf.nn.conv2d(h_conv6, W_conv7, strides= [1,1,1,1], padding= 'SAME',
     dilations= [1, 1, 1, 1], name='dilation7')
    h_conv7 = tf.nn.relu(c_dilat7 + b_conv7)

    #1*1 layer
    W_conv8 = weight_variable([1,1,32,10])
    b_conv8 = bias_variable([10])
    c_dilat8 = tf.nn.conv2d(h_conv7, W_conv8,strides= [1,1,1,1], padding= 'SAME',
     dilations= [1,1,1,1], name= 'dilation8')
    flat = flatten(c_dilat8)

    fc1 = fully_conn(flat,512)
    fc1 = tf.nn.dropout(fc1,keep_prob)
    
    out = output(fc1, 10)	
    l2_loss = tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_conv3)+tf.nn.l2_loss(W_conv4)+tf.nn.l2_loss(W_conv5)+tf.nn.l2_loss(W_conv6)+tf.nn.l2_loss(W_conv7)+tf.nn.l2_loss(W_conv8)
    return out,l2_loss


