import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

epochs_completed = 0
index_in_epoch = 0

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

def one_hot_encode(x):
	encoder = preprocessing.LabelBinarizer()
	list1 = np.arange(0,100)
	encoder.fit(list1)
	x = encoder.transform(x)
	return x

def next_batch(batch_size):

	global train_Data
	global train_label
	global index_in_epoch
	global epochs_completed

	start = index_in_epoch
	index_in_epoch += batch_size

	# when all trainig data have been already used, it is reorder randomly
	if index_in_epoch > num_examples:
		# finished epoch
		epochs_completed += 1
		# shuffle the data
		perm = np.arange(num_examples)
		np.random.shuffle(perm)
		train_Data = [train_Data[i] for i in perm]
		train_label = [train_label[i] for i in perm]
		# start next epoch
		start = 0
		index_in_epoch = batch_size
		assert batch_size <= num_examples
	end = index_in_epoch
	return train_Data[start:end], train_label[start:end]

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
