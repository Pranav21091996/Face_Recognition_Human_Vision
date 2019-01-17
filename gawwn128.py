import os
import sys
import numpy as np
import datetime
import dateutil.tz
import argparse
from shutil import copyfile
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.layers import utils
import tensorflow.contrib.eager as tfe
from tensorflow.python.client import timeline
import cv2
import scipy.misc
import PIL
import matplotlib.patches as patches
from PIL import Image

image = np.load('drive/image128.npy')
label = np.load('drive/labels128.npy')
box = np.load('drive/bbox128.npy')

Data = image
label = label
bbox = box
epochs_completed = 0
index_in_epoch = 0
num_examples = len(image)

def next_batch(BATCH_SIZE):

  global Data
  global label
  global bbox
  global index_in_epoch
  global epochs_completed

  start = index_in_epoch
  index_in_epoch += BATCH_SIZE

	# when all trainig data have been already used, it is reorder randomly
  if index_in_epoch > num_examples:
		# finished epoch
    epochs_completed += 1
		# shuffle the data
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    Data = [Data[i] for i in perm]
    label = [label[i] for i in perm]
    bbox = [bbox[i] for i in perm]
		# start next epoch
    start = 0
    index_in_epoch = BATCH_SIZE
    assert BATCH_SIZE <= num_examples
  end = index_in_epoch
  return Data[start:end], label[start:end], bbox[start:end]

BATCH_SIZE = 32     # training batch size
batch_size = 32
IMG_WIDTH, IMG_HEIGHT = 128, 128      # image dimensions
IMG_CHANNELS = 3                    # image channels
n_batches = int(num_examples/BATCH_SIZE)
Z_DIM = 128                # dimensionality of the z vector (input to G, incompressible noise)
LABEL_DIM = 200                      # dimensionality of the label vector (axis 1)
LOGDIR = 'drive/logs'
weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

def sample_z(m, n):
    return np.random.normal(loc=0.0, scale=1.0, size=(m, n)).astype(np.float32)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = y.get_shape()
    y = tf.reshape(y, (x_shapes[0], 1, 1, y_shapes[1]))
    z = tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[1]])
    y = tf.cast(y,'float32')
    y = y*z

    return tf.concat([x , y], 3)

def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):
    """
    Spatial Transformer Network layer implementation as described in [1].
    The layer is composed of 3 elements:
    - localisation_net: takes the original image as input and outputs 
      the parameters of the affine transformation that should be applied
      to the input image.
    - affine_grid_generator: generates a grid of (x,y) coordinates that 
      correspond to a set of points where the input should be sampled 
      to produce the transformed output.
    - bilinear_sampler: takes as input the original image and the grid
      and produces the output image using bilinear interpolation.
    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
      transformer layer is at the beginning of architecture. Should be 
      a tensor of shape (B, H, W, C). 
    - theta: affine transform tensor of shape (B, 6). Permits cropping, 
      translation and isotropic scaling. Initialize to identity matrix. 
      It is the output of the localization network.
    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)
    """
    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]
    C = tf.shape(input_fmap)[3]

    # reshape theta to (B, 2, 3)
    theta = tf.reshape(theta, [B, 2, 3])

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

    return out_fmap

def tf_compute_transformation_matrix(bbox, shape=16., img_height=64):
    rel_factor = float(shape)/img_height
    x, y, w, h = rel_factor*bbox[0], rel_factor*bbox[1], rel_factor*bbox[2], rel_factor*bbox[3]

    t_x = (x+0.5*w-0.5*shape)/(0.5*shape)
    t_y = (y+0.5*h-0.5*shape)/(0.5*shape)

    scale_x = (w / shape)
    scale_y = (h / shape)

    line0 = tf.stack((scale_x, 0.0, t_x))
    line1 = tf.stack((0.0, scale_y, t_y))
    transformation_matrix = tf.concat((line0, line1), axis=0)
    transformation_matrix = tf.reshape(transformation_matrix, (2, 3))

    return transformation_matrix 
  
  
def tf_compute_transformation_matrix_inverse(bbox, shape=16., img_height=64):
    rel_factor = float(shape)/img_height
    x, y, w, h = rel_factor*bbox[0], rel_factor*bbox[1], rel_factor*bbox[2], rel_factor*bbox[3]

    scale_x = (shape / w)
    scale_y = (shape / h)

    t_x = (shape - 2 * x) / w - 1
    t_y = (shape - 2 * y) / h - 1

    line0 = tf.stack((scale_x, 0.0, t_x))
    line1 = tf.stack((0.0, scale_y, t_y))

    transformation_matrix = tf.concat((line0, line1), axis=0)
    transformation_matrix = tf.reshape(transformation_matrix, (2, 3))

    return transformation_matrix  
  
def sample_gen_label(mb_size):
    labels = np.random.multinomial(1, 200*[1/200], size=mb_size)
    return labels.astype(np.float32)


def sample_gen_label_sorted(mb_size, label_dim=10):
    labels = np.zeros((mb_size, label_dim))
    for idx in range(label_dim):
        labels[idx*label_dim:idx*label_dim+label_dim, idx] = 1
    return labels.astype(np.float32)


def sample_bbox(mb_size):
    pos_box_x = np.random.randint(low=0, high=44, size=(mb_size, 1))
    pos_box_y = np.random.randint(low=0, high=44, size=(mb_size, 1))

    coin = np.random.binomial(1, 0.5)
    if coin < 0.1:
        scale_box_x = np.random.randint(low=8, high=16, size=(mb_size, 1))
    elif 0.1 < coin < 0.8:
        scale_box_x = np.random.randint(low=12, high=18, size=(mb_size, 1))
    else:
        scale_box_x = np.random.randint(low=16, high=21, size=(mb_size, 1))
    scale_box_y = np.random.randint(low=18, high=21, size=(mb_size, 1))

    boxes = np.concatenate((pos_box_x, pos_box_y, scale_box_x, scale_box_y), axis=1)
    return boxes.astype(np.float32)


def sample_bbox_sorted(mb_size, sort_size=False):
    pos_box_x = np.zeros((mb_size, 1))
    for idx in range(100):
        pos_box_x[idx, 0] = (idx % 10 + 1) * 4

    pos_box_y = np.zeros((mb_size, 1))
    for idx in range(10):
        pos_box_y[idx * 10:idx * 10 + 10, 0] = (idx + 1) * 4

    if sort_size:
        scale_box_x = np.zeros((mb_size, 1))
        for idx in range(100):
            scale_box_x[idx, 0] = (idx % 10 + 1) + 10

        scale_box_y = np.zeros((mb_size, 1))
        for idx in range(10):
            scale_box_y[idx * 10:idx * 10 + 10, 0] = (idx + 1) + 10
    else:
        scale_box_x = np.random.randint(low=16, high=21, size=(mb_size, 1))
        scale_box_y = np.random.randint(low=18, high=21, size=(mb_size, 1))

    boxes = np.concatenate((pos_box_x, pos_box_y, scale_box_x, scale_box_y), axis=1)
    return boxes.astype(np.float32)


def sample_generator_input(mb_size, n, sort_labels=False, sort_location=False, sort_bbox_size=False):
    _z = sample_z(mb_size, n)
    _Y = sample_gen_label_sorted(mb_size) if sort_labels else sample_gen_label(mb_size)

    if sort_bbox_size:
        _bbox = sample_bbox_sorted(mb_size, sort_size=True)
    else:
        _bbox = sample_bbox_sorted(mb_size) if sort_location else sample_bbox(mb_size)


    return _z, _Y, _bbox

def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature 
    map, will create an output feature map that is an 
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample. 
    - width: desired width of grid/output. Used
      to downsample or upsample. 
    - theta: affine transform matrices of shape (num_batch, 2, 3). 
      For each image in the batch, we have 6 theta parameters of 
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized gird (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the 
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation, 
         and isotropic scaling.
    """
    # grab batch size
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the 
    normalized coordinates provided by the sampling grid. Note that 
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - interpolated images according to grids. Same size as grid.
    """
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # rescale x and y to [0, W/H]
    x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])


    return out  
  
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    BATCH_SIZE = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, BATCH_SIZE)
    batch_idx = tf.reshape(batch_idx, (BATCH_SIZE, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def transform(img):
    img = np.divide(np.add(img,1),2.0)
    return img.astype(np.float32)
  
def process_oneimg(img):
  img = np.subtract(np.multiply(img , 2),1.0)
  return img

def lrelu(x,alpha=0.2):
	return tf.maximum(x,alpha*x)	

def conv2d(x, features, kernel=[4,4], strides=[1,2,2,1], name="conv_layer"):
	with tf.variable_scope(name) as scope:
		weights = weight(shape=kernel + features, name="weights")
		biases = bias(shape=[features[-1]], name="bias")
		output = tf.nn.conv2d(x, weights, strides=strides, padding='SAME') 
		output = tf.nn.bias_add(output, biases)
		return output	

def deconv2d(x, features, output_shape, kernel=[4,4], strides=[1,2,2,1], name="deconv_layer"):
	with tf.variable_scope(name) as scope:
		weights = weight(shape=kernel + features, name="weights")
		biases = bias(shape=[features[0]], name="bias")
		output = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape, strides=strides, padding='SAME') 
		return tf.reshape(tf.nn.bias_add(output, biases), output.get_shape())

def bias(shape, name):
	return tf.get_variable(name, shape,initializer=tf.constant_initializer(0.00000))

def weight(shape, name):
	return tf.get_variable(name, shape,initializer=tf.glorot_uniform_initializer())	

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def attention(x, ch, sn=False, scope='attention', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    f = conv2d(x, ch // 8, kernel=1, stride=1, sn=sn, name='f_conv') # [bs, h, w, c']
    g = conv2d(x, ch // 8, kernel=1, stride=1, sn=sn, name='g_conv') # [bs, h, w, c']
    h = conv2d(x, ch, kernel=1, stride=1, sn=sn, name='h_conv') # [bs, h, w, c]
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]
    beta = tf.nn.softmax(s)  # attention map
    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
    o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
    x = gamma * o + x
  return x  

def dense(x, shape, name):
	with tf.variable_scope(name):
		weights = weight(shape, name="weights")
		biases = bias([shape[-1]], name="bias")
		return tf.matmul(x,weights) + biases

def hw_flatten(x) :
    dim1 = tf.shape(x)[0]
    dim2 = tf.shape(x)[-1]
    return tf.reshape(x, shape=[dim1, -1, dim2])  
  
def batch_norm(inputs, decay=0.9, epsilon=0.00001, scale=True, isTrain=True, name="batch_norm"):
	return tf.contrib.layers.batch_norm(inputs, decay=decay, scale=scale, epsilon=epsilon, updates_collections=None, is_training=isTrain, scope=name)

def discriminate(image_input, bounding_box, input_labels, reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()
    input = conv_cond_concat(image_input, input_labels)
    d_x_conv_0 = conv2d(input, features=[203, 64],kernel=[5,5], name="d_conv_layer_1")
    d_x_conv_0 = lrelu(d_x_conv_0)
    
    d_x_conv_1 = conv2d(d_x_conv_0, features=[64, 128], kernel=[5,5],name="d_conv_layer_2")
    d_x_conv_1 = batch_norm(d_x_conv_1, isTrain=True, name="d_batch_norm_2")
    d_x_conv_1 = lrelu(d_x_conv_1)
    
    d_x_conv_2 = conv2d(d_x_conv_1, features=[128, 256], kernel=[5,5],name="d_conv_layer_3")
    d_x_conv_2 = batch_norm(d_x_conv_2, isTrain=True, name="d_batch_norm_3")
    d_x_conv_2 = lrelu(d_x_conv_2)
      
    ####################################################
    # global pathway
      
    d_x_conv_global_0 = conv2d(d_x_conv_2, features=[256, 512], kernel=[5,5],name="d_conv_global_1")
    d_x_conv_global_0 = batch_norm(d_x_conv_global_0, isTrain=True, name="d_batch_global_1")
    d_x_conv_global_0 = lrelu(d_x_conv_global_0)
      
    d_x_conv_global_1 = conv2d(d_x_conv_global_0, features=[512, 1024], kernel=[5,5],name="d_conv_global_2")
    d_x_conv_global_1 = batch_norm(d_x_conv_global_1, isTrain=True, name="d_batch_global_2")
    d_x_conv_global_1 = lrelu(d_x_conv_global_1)
    
    shp = [int(s) for s in d_x_conv_global_1.shape[1:]]
    d_x_conv_global_1 = tf.reshape(d_x_conv_global_1, [-1, shp[0] * shp[1] * shp[2]])

    ####################################################
    # local pathway
    # reshape bounding box to (16, 16) resolution
    transf_matri = tf.map_fn(tf_compute_transformation_matrix, bounding_box)
    local_input = spatial_transformer_network(d_x_conv_2, transf_matri, (16, 16))
      
    d_x_conv_local_0 = conv2d(local_input, features=[256, 512], kernel=[5,5],name="d_conv_local_0")
    d_x_conv_local_0 = batch_norm(d_x_conv_local_0, isTrain=True, name="d_batch_local_0")
    d_x_conv_local_0 = lrelu(d_x_conv_local_0)
      
    d_x_conv_local_1 = conv2d(d_x_conv_local_0, features=[512, 1024], kernel=[5,5],name="d_conv_local_1")
    d_x_conv_local_1 = batch_norm(d_x_conv_local_1, isTrain=True, name="d_batch_local_1")
    d_x_conv_local_1 = lrelu(d_x_conv_local_1)
    
    shp = [int(s) for s in d_x_conv_local_1.shape[1:]]
    d_x_conv_local_1 = tf.reshape(d_x_conv_local_1, [-1, shp[0] * shp[1] * shp[2]])

    ####################################################
    # final discriminator
    final_input = tf.concat((d_x_conv_global_1, d_x_conv_local_1), axis=1)
    dimension = final_input.get_shape().as_list()[1]
    d_final_dense = dense(final_input, shape=[dimension, 512], name="d_final_dense_1")
    d_final_dense = batch_norm(d_final_dense, isTrain=True, name="d_batch_dense_1")
    d_final_dense = lrelu(d_final_dense)
          
    d_final_pred = dense(d_final_dense, shape=[512, 1],name="d_final_dense_2")

    return d_final_pred,tf.nn.sigmoid(d_final_pred)

def sampler(noise_input, bounding_box, input_labels):
  with tf.variable_scope("g_net") as scope:
    scope.reuse_variables()
    
    input = tf.concat((noise_input, input_labels), axis=1)
    g_dense_0 = dense(input, shape=[328, 1024], name="g_dense_1")
    g_dense_0 = batch_norm(g_dense_0, isTrain=False, name="g_batch_norm_0")
    g_dense_0 = tf.nn.relu(g_dense_0)
    g_dense_0 = tf.reshape(g_dense_0, [-1, 2, 2, 256])
    
    ####################################################
    # global pathway
        
    g_conv_global_0 = deconv2d(g_dense_0, features=[512,256], output_shape=[batch_size,4,4,512], name="g_deconv_global_1")
    g_conv_global_0 = batch_norm(g_conv_global_0, isTrain=False, name="g_batch_global_1")
    g_conv_global_0 = tf.nn.relu(g_conv_global_0)

    g_conv_global_1 = deconv2d(g_conv_global_0, features=[256,512], output_shape=[batch_size,8,8,256], name="g_deconv_global_2")
    g_conv_global_1 = batch_norm(g_conv_global_1, isTrain=False, name="g_batch_global_2")
    g_conv_global_1 = tf.nn.relu(g_conv_global_1)
    
    g_conv_global_2 = deconv2d(g_conv_global_1, features=[128,256], output_shape=[batch_size,16,16,128], name="g_deconv_global_3")
    g_conv_global_2 = batch_norm(g_conv_global_2, isTrain=False, name="g_batch_global_3")
    g_conv_global_2 = tf.nn.relu(g_conv_global_2)
            
    ####################################################
    # local pathway
        
    g_conv_local_0 = deconv2d(g_dense_0, features=[512,256], output_shape=[batch_size,4,4,512], name="g_deconv_local_1")
    g_conv_local_0 = batch_norm(g_conv_local_0, isTrain=False, name="g_batch_local_1")
    g_conv_local_0 = tf.nn.relu(g_conv_local_0)

    g_conv_local_1 = deconv2d(g_conv_local_0, features=[256,512], output_shape=[batch_size,8,8,256], name="g_deconv_local_2")
    g_conv_local_1 = batch_norm(g_conv_local_1, isTrain=False, name="g_batch_local_2")
    g_conv_local_1 = tf.nn.relu(g_conv_local_1)
    
    g_conv_local_2 = deconv2d(g_conv_local_1, features=[128,256], output_shape=[batch_size,16,16,128], name="g_deconv_local_3")
    g_conv_local_2 = batch_norm(g_conv_local_2, isTrain=False, name="g_batch_local_3")
    g_conv_local_2 = tf.nn.relu(g_conv_local_2)
    
    
    # reshape to bounding box
    transf_matri = tf.map_fn(tf_compute_transformation_matrix_inverse, bounding_box)
    g_conv_local_2 = spatial_transformer_network(g_conv_local_2, transf_matri, (16, 16))

    ####################################################
    # final pathway
    final_input = tf.concat((g_conv_global_2, g_conv_local_2), axis=3)
    
    g_conv_final_0 = deconv2d(final_input, features=[128,256], output_shape=[batch_size,32,32,128], name="g_deconv_final_0")
    g_conv_final_0 = batch_norm(g_conv_final_0, isTrain=False, name="g_batch_final_0")
    g_conv_final_0 = tf.nn.relu(g_conv_final_0)
    
    #g_conv_final_0 = attention(g_conv_final_0, 64, scope="attention", reuse=reuse)
    
    g_conv_final_1 = deconv2d(g_conv_final_0, features=[64,128], output_shape=[batch_size,64,64,64],  name="g_deconv_final_1")
    g_conv_final_1 = batch_norm(g_conv_final_1, isTrain=False, name="g_batch_final_1")
    g_conv_final_1 = tf.nn.relu(g_conv_final_1)
        
    g_conv_final_2 = deconv2d(g_conv_final_1, features=[3,64], output_shape=[batch_size,128,128,3],  name="g_deconv_final_2")
    g_conv_out= tf.nn.tanh(g_conv_final_2)

    return g_conv_out

def generate(noise_input, bounding_box, input_labels):
  with tf.variable_scope("g_net") as scope:
    
    input = tf.concat((noise_input, input_labels), axis=1)
    g_dense_0 = dense(input, shape=[328, 1024], name="g_dense_1")
    g_dense_0 = batch_norm(g_dense_0, isTrain=True, name="g_batch_norm_0")
    g_dense_0 = tf.nn.relu(g_dense_0)
    g_dense_0 = tf.reshape(g_dense_0, [-1, 2, 2, 256])
    
    ####################################################
    # global pathway
        
    g_conv_global_0 = deconv2d(g_dense_0, features=[512,256], output_shape=[batch_size,4,4,512], name="g_deconv_global_1")
    g_conv_global_0 = batch_norm(g_conv_global_0, isTrain=True, name="g_batch_global_1")
    g_conv_global_0 = tf.nn.relu(g_conv_global_0)

    g_conv_global_1 = deconv2d(g_conv_global_0, features=[256,512], output_shape=[batch_size,8,8,256], name="g_deconv_global_2")
    g_conv_global_1 = batch_norm(g_conv_global_1, isTrain=True, name="g_batch_global_2")
    g_conv_global_1 = tf.nn.relu(g_conv_global_1)
    
    g_conv_global_2 = deconv2d(g_conv_global_1, features=[128,256], output_shape=[batch_size,16,16,128], name="g_deconv_global_3")
    g_conv_global_2 = batch_norm(g_conv_global_2, isTrain=True, name="g_batch_global_3")
    g_conv_global_2 = tf.nn.relu(g_conv_global_2)
            
    ####################################################
    # local pathway
        
    g_conv_local_0 = deconv2d(g_dense_0, features=[512,256], output_shape=[batch_size,4,4,512], name="g_deconv_local_1")
    g_conv_local_0 = batch_norm(g_conv_local_0, isTrain=True, name="g_batch_local_1")
    g_conv_local_0 = tf.nn.relu(g_conv_local_0)

    g_conv_local_1 = deconv2d(g_conv_local_0, features=[256,512], output_shape=[batch_size,8,8,256], name="g_deconv_local_2")
    g_conv_local_1 = batch_norm(g_conv_local_1, isTrain=True, name="g_batch_local_2")
    g_conv_local_1 = tf.nn.relu(g_conv_local_1)
    
    g_conv_local_2 = deconv2d(g_conv_local_1, features=[128,256], output_shape=[batch_size,16,16,128], name="g_deconv_local_3")
    g_conv_local_2 = batch_norm(g_conv_local_2, isTrain=True, name="g_batch_local_3")
    g_conv_local_2 = tf.nn.relu(g_conv_local_2)
    
    
    # reshape to bounding box
    transf_matri = tf.map_fn(tf_compute_transformation_matrix_inverse, bounding_box)
    g_conv_local_2 = spatial_transformer_network(g_conv_local_2, transf_matri, (16, 16))

    ####################################################
    # final pathway
    final_input = tf.concat((g_conv_global_2, g_conv_local_2), axis=3)
    
    g_conv_final_0 = deconv2d(final_input, features=[128,256], output_shape=[batch_size,32,32,128], name="g_deconv_final_0")
    g_conv_final_0 = batch_norm(g_conv_final_0, isTrain=True, name="g_batch_final_0")
    g_conv_final_0 = tf.nn.relu(g_conv_final_0)
    
    #g_conv_final_0 = attention(g_conv_final_0, 64, scope="attention", reuse=reuse)
    
    g_conv_final_1 = deconv2d(g_conv_final_0, features=[64,128], output_shape=[batch_size,64,64,64],  name="g_deconv_final_1")
    g_conv_final_1 = batch_norm(g_conv_final_1, isTrain=True, name="g_batch_final_1")
    g_conv_final_1 = tf.nn.relu(g_conv_final_1)
        
    g_conv_final_2 = deconv2d(g_conv_final_1, features=[3,64], output_shape=[batch_size,128,128,3],  name="g_deconv_final_2")
    g_conv_out= tf.nn.tanh(g_conv_final_2)

    return g_conv_out

tf.reset_default_graph()

Y = tf.placeholder(tf.float32, shape=[BATCH_SIZE,LABEL_DIM], name='label')
X = tf.placeholder(tf.float32, shape=[BATCH_SIZE,IMG_WIDTH,IMG_WIDTH,3], name='image')
b = tf.placeholder(tf.float32, shape=[BATCH_SIZE,4], name='box')

z = tf.placeholder(tf.float32, shape=[BATCH_SIZE,Z_DIM], name='z')
Y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE,LABEL_DIM], name='label_gen')
b_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE,4], name='box_gen')

real_img_real_label_disc_logits, real_disc_real = discriminate(image_input=X, bounding_box=b, input_labels=Y)
fake_images = generate(noise_input=z, bounding_box=b, input_labels=Y)
fake_disc_logits, fake_disc = discriminate(image_input=fake_images, bounding_box=b, input_labels=Y, reuse=True)
real_img_fake_label_disc_logits, real_disc_fake = discriminate(image_input=X, bounding_box=b_, input_labels=Y_,reuse=True)
sample = sampler(noise_input=z, bounding_box=b, input_labels=Y)

d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_img_real_label_disc_logits), logits=real_img_real_label_disc_logits))
d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_disc_logits), logits=fake_disc_logits))
d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_img_fake_label_disc_logits), logits=real_img_fake_label_disc_logits))
d_loss = d_loss1 + d_loss2 + d_loss3
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_disc_logits), logits=fake_disc_logits))

tf.summary.scalar("g_loss", g_loss)
tf.summary.scalar("d_loss", d_loss)

d_learning_rate = 0.0004
g_learning_rate = 0.0001
beta1 = 0.0
beta2 = 0.9

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_optim = tf.train.AdamOptimizer(d_learning_rate, beta1=beta1, beta2=beta2).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(g_learning_rate, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(max_to_keep=2)
  summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)
  ckpt = tf.train.get_checkpoint_state(LOGDIR)
  epoch=0
  
  if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("Model restored...")
      epoch = int(ckpt.model_checkpoint_path.split('-')[1])
      print("starting from iteration", epoch)

  while epoch!=10:
    dl, gl = [],[]
       
    
    for i in range(10000):
      print(i,"th iteration")
      z_mb, labe, box_generated = sample_generator_input(BATCH_SIZE, Z_DIM)
      data_1,  labe_1, bbox_1 = next_batch(BATCH_SIZE)
      data_1 = process_oneimg(data_1)
      feed_dict = {X:data_1,b:bbox_1,z:z_mb,b_:box_generated,Y:labe_1,Y_:labe}
      
      _, DLOSS = sess.run([d_optim, d_loss],feed_dict=feed_dict)
      _, GLOSS = sess.run([g_optim, g_loss],feed_dict=feed_dict)
      
      dl.append(DLOSS)
      gl.append(GLOSS)
      if i % 500 == 0 and i > 0 :
        
        print('discriminator_loss / generator_loss => %.2f / %.2f for iteration of epoch %d'%(np.mean(dl), np.mean(gl), epoch))
        z_mb, labe, box_generated = sample_generator_input(BATCH_SIZE, Z_DIM)
        data_1, labe_1, bbox_1 = next_batch(BATCH_SIZE)
        data_1 = process_oneimg(data_1)
        feed_dict = {X:data_1,b:bbox_1,z:z_mb,b_:box_generated,Y:labe_1,Y_:labe}
        fake_image = sess.run(sample,feed_dict=feed_dict)
        plt.imsave('drive/new_result/'+str(epoch)+'_'+str(i)+'.jpg',transform(fake_image[0]))
    saver.save(sess, LOGDIR + "/model.ckpt", global_step=epoch)
    epoch+=1


