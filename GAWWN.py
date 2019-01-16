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
import tensorlayer as tl
import cv2
import scipy.misc
import PIL
import matplotlib.patches as patches
from PIL import Image
from utils2 import *


image = np.load('drive/image.npy')
label = np.load('drive/labels.npy')
box = np.load('drive/bbox.npy')


Data = image
label = label
bbox = box
epochs_completed = 0
index_in_epoch = 0
num_examples = len(image)

def next_batch(batch_size):

    global Data
    global label
    global bbox
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
        Data = [Data[i] for i in perm]
        label = [label[i] for i in perm]
        bbox = [bbox[i] for i in perm]
		# start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return Data[start:end], label[start:end], bbox[start:end]


BATCH_SIZE = 16        # training batch size
batch_size = 16
IMG_WIDTH, IMG_HEIGHT = 64, 64      # image dimensions
IMG_CHANNELS = 3                    # image channels
n_batches = int(num_examples/batch_size)
Z_DIM = 128                # dimensionality of the z vector (input to G, incompressible noise)
LABEL_DIM = 200                      # dimensionality of the label vector (axis 1)
weight_init = tf.truncated_normal_initializer(stddev=0.02)
LOGDIR = 'logs'


def discriminate(image_input, label, bounding_box,reuse):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        input = conv_cond_concat(image_input, label)
        d_x_conv_0 = conv2d(input, features=[3, 32],kernel=[5,5], name="d_conv_layer_1")
        d_x_conv_0 = lrelu(d_x_conv_0)
      
        d_x_conv_1 = conv2d(d_x_conv_0, features=[32, 64], kernel=[5,5],name="d_conv_layer_2")
        d_x_conv_1 = batch_norm(d_x_conv_1, isTrain=True, name="d_batch_norm_2")
        d_x_conv_1 = lrelu(d_x_conv_1)
      
    ####################################################
    # global pathway
      
        d_x_conv_global_0 = conv2d(d_x_conv_1, features=[64, 64], kernel=[5,5],name="d_conv_global_1")
        d_x_conv_global_0 = batch_norm(d_x_conv_global_0, isTrain=True, name="d_batch_global_1")
        d_x_conv_global_0 = lrelu(d_x_conv_global_0)
      
        d_x_conv_global_1 = conv2d(d_x_conv_global_0, features=[64, 128], kernel=[5,5],name="d_conv_global_2")
        d_x_conv_global_1 = batch_norm(d_x_conv_global_1, isTrain=True, name="d_batch_global_2")
        d_x_conv_global_1 = lrelu(d_x_conv_global_1)
      
        shp = [int(s) for s in d_x_conv_global_1.shape[1:]]
        d_x_conv_global_1 = tf.reshape(d_x_conv_global_1, [-1, shp[0] * shp[1] * shp[2]])

    ####################################################
    # local pathway
    # reshape bounding box to (16, 16) resolution
        transf_matri = tf.map_fn(tf_compute_transformation_matrix, bounding_box)
        local_input = spatial_transformer_network(d_x_conv_1, transf_matri, (16, 16))
      
        d_x_conv_local_0 = conv2d(local_input, features=[64, 64], kernel=[5,5],name="d_conv_local_0")
        d_x_conv_local_0 = batch_norm(d_x_conv_local_0, isTrain=True, name="d_batch_local_0")
        d_x_conv_local_0 = lrelu(d_x_conv_local_0)
      
        d_x_conv_local_1 = conv2d(d_x_conv_local_0, features=[64, 128], kernel=[5,5],name="d_conv_local_1")
        d_x_conv_local_1 = batch_norm(d_x_conv_local_1, isTrain=True, name="d_batch_local_1")
        d_x_conv_local_1 = lrelu(d_x_conv_local_1)
      
        shp = [int(s) for s in d_x_conv_local_1.shape[1:]]
        d_x_conv_local_1 = tf.reshape(d_x_conv_local_1, [-1, shp[0] * shp[1] * shp[2]])

    ####################################################
    # final discriminator
        final_input = tf.concat((d_x_conv_global_1, d_x_conv_local_1), axis=1)
      
        d_final_dense = dense(final_input, shape=[4096, 512], name="d_final_dense_1")
        d_final_dense = batch_norm(d_final_dense, isTrain=True, name="d_batch_dense_1")
        d_final_dense = lrelu(d_final_dense)
      
        d_final_pred = dense(d_final_dense, shape=[512, 1],name="d_final_dense_2")

        return d_final_pred,tf.nn.sigmoid(d_final_pred)

def sampler(noise_input, label, bounding_box):
    with tf.variable_scope("g_net") as scope:
        scope.reuse_variables()
    
        input = tf.concat((noise_input, label), axis=1)
        g_dense_0 = dense(input, shape=[128, 2048], name="g_dense_1")
        g_dense_0 = batch_norm(g_dense_0, isTrain=False, name="g_batch_norm_0")
        g_dense_0 = tf.nn.relu(g_dense_0)

        g_dense_0 = tf.reshape(g_dense_0, [-1, 4, 4, 128])

    ####################################################
    # global pathway
        
        g_conv_global_0 = deconv2d(g_dense_0, features=[256, 128], output_shape=[batch_size,8,8,256], name="g_deconv_global_1")
        g_conv_global_0 = batch_norm(g_conv_global_0, isTrain=False, name="g_batch_global_1")
        g_conv_global_0 = tf.nn.relu(g_conv_global_0)

        g_conv_global_1 = deconv2d(g_conv_global_0, features=[256, 256], output_shape=[batch_size,16,16,256], name="g_deconv_global_2")
        g_conv_global_1 = batch_norm(g_conv_global_1, isTrain=False, name="g_batch_global_2")
        g_conv_global_1 = tf.nn.relu(g_conv_global_1)
        
        
    ####################################################
    # local pathway
        
        g_conv_local_0 = deconv2d(g_dense_0, features=[256, 128], output_shape=[batch_size,8,8,256], name="g_deconv_local_1")
        g_conv_local_0 = batch_norm(g_conv_local_0, isTrain=False, name="g_batch_local_1")
        g_conv_local_0 = tf.nn.relu(g_conv_local_0)

        g_conv_local_1 = deconv2d(g_conv_local_0, features=[256, 256], output_shape=[batch_size,16,16,256], name="g_deconv_local_2")
        g_conv_local_1 = batch_norm(g_conv_local_1, isTrain=False, name="g_batch_local_2")
        g_conv_local_1 = tf.nn.relu(g_conv_local_1)
        
    # reshape to bounding box
        transf_matri = tf.map_fn(tf_compute_transformation_matrix_inverse, bounding_box)
        g_conv_local_1 = spatial_transformer_network(g_conv_local_1, transf_matri, (16, 16))

    ####################################################
    # final pathway
        final_input = tf.concat((g_conv_global_1, g_conv_local_1), axis=3)
        g_conv_final = deconv2d(final_input, features=[256, 512], output_shape=[batch_size,32,32,256], name="g_deconv_final_1")
        g_conv_final = batch_norm(g_conv_final, isTrain=False, name="g_batch_final_1")
        g_conv_final = tf.nn.relu(g_conv_final)
        
        g_conv_final_2 = deconv2d(g_conv_final, features=[256, 256], output_shape=[batch_size,64,64,256], name="g_deconv_final_2")
        g_conv_final_2 = batch_norm(g_conv_final_2, isTrain=False, name="g_batch_final_2")
        g_conv_final_2 = tf.nn.relu(g_conv_final_2)
        
        
        g_conv_out = deconv2d(g_conv_final_2, features=[3, 256], output_shape=[batch_size,64,64,3],strides=[1,1,1,1], name="g_deconv_output_1")
        g_conv_out= tf.nn.tanh(g_conv_out)

        return g_conv_out


def generate(noise_input, label, bounding_box):
    with tf.variable_scope("g_net") as scope:
        input = tf.concat((noise_input, label), axis=1)
        g_dense_0 = dense(input, shape=[128, 2048], name="g_dense_1")
        g_dense_0 = batch_norm(g_dense_0, isTrain=True, name="g_batch_norm_0")
        g_dense_0 = tf.nn.relu(g_dense_0)

        g_dense_0 = tf.reshape(g_dense_0, [-1, 4, 4, 128])

    ####################################################
    # global pathway
        
        g_conv_global_0 = deconv2d(g_dense_0, features=[256, 128], output_shape=[batch_size,8,8,256], name="g_deconv_global_1")
        g_conv_global_0 = batch_norm(g_conv_global_0, isTrain=True, name="g_batch_global_1")
        g_conv_global_0 = tf.nn.relu(g_conv_global_0)

        g_conv_global_1 = deconv2d(g_conv_global_0, features=[256, 256], output_shape=[batch_size,16,16,256], name="g_deconv_global_2")
        g_conv_global_1 = batch_norm(g_conv_global_1, isTrain=True, name="g_batch_global_2")
        g_conv_global_1 = tf.nn.relu(g_conv_global_1)
        
        
    ####################################################
    # local pathway
        
        g_conv_local_0 = deconv2d(g_dense_0, features=[256, 128], output_shape=[batch_size,8,8,256], name="g_deconv_local_1")
        g_conv_local_0 = batch_norm(g_conv_local_0, isTrain=True, name="g_batch_local_1")
        g_conv_local_0 = tf.nn.relu(g_conv_local_0)

        g_conv_local_1 = deconv2d(g_conv_local_0, features=[256, 256], output_shape=[batch_size,16,16,256], name="g_deconv_local_2")
        g_conv_local_1 = batch_norm(g_conv_local_1, isTrain=True, name="g_batch_local_2")
        g_conv_local_1 = tf.nn.relu(g_conv_local_1)
        
    # reshape to bounding box
        transf_matri = tf.map_fn(tf_compute_transformation_matrix_inverse, bounding_box)
        g_conv_local_1 = spatial_transformer_network(g_conv_local_1, transf_matri, (16, 16))

    ####################################################
    # final pathway
        final_input = tf.concat((g_conv_global_1, g_conv_local_1), axis=3)
        g_conv_final = deconv2d(final_input, features=[256, 512], output_shape=[batch_size,32,32,256], name="g_deconv_final_1")
        g_conv_final = batch_norm(g_conv_final, isTrain=True, name="g_batch_final_1")
        g_conv_final = tf.nn.relu(g_conv_final)
        
        g_conv_final_2 = deconv2d(g_conv_final, features=[256, 256], output_shape=[batch_size,64,64,256], name="g_deconv_final_2")
        g_conv_final_2 = batch_norm(g_conv_final_2, isTrain=True, name="g_batch_final_2")
        g_conv_final_2 = tf.nn.relu(g_conv_final_2)
        
        
        g_conv_out = deconv2d(g_conv_final_2, features=[3, 256], output_shape=[batch_size,64,64,3],strides=[1,1,1,1], name="g_deconv_output_1")
        g_conv_out= tf.nn.tanh(g_conv_out)

        return g_conv_out


tf.reset_default_graph()

Y = tf.placeholder(tf.float32, shape=[None,LABEL_DIM], name='label')
X = tf.placeholder(tf.float32, shape=[None,64,64,3], name='image')
b = tf.placeholder(tf.float32, shape=[None,4], name='box')

z = tf.placeholder(tf.float32, shape=[None,Z_DIM], name='z')
Y_ = tf.placeholder(tf.float32, shape=[None,10], name='label_gen')
b_ = tf.placeholder(tf.float32, shape=[None,4], name='box_gen')

fake_images = generate(noise_input=z, label=Y, bounding_box=b)
fake_disc_logits, fake_disc = discriminate(image_input=fake_images, label=Y, bounding_box=b, reuse=False)
real_img_real_label_disc_logits, real_disc_real = discriminate(image_input=X, label=Y, bounding_box=b, reuse=True)
real_img_fake_label_disc_logits, real_disc_fake = discriminate(image_input=X, label=Y_, bounding_box=b_, reuse=True)
sample = sampler(noise_input=z, label=Y, bounding_box=b)


d_loss1 = tl.cost.sigmoid_cross_entropy(real_img_real_label_disc_logits, tf.ones_like(real_img_real_label_disc_logits), name='d1')
d_loss2 = tl.cost.sigmoid_cross_entropy(real_img_fake_label_disc_logits,  tf.zeros_like(real_img_fake_label_disc_logits), name='d2')
d_loss3 = tl.cost.sigmoid_cross_entropy(fake_disc_logits, tf.zeros_like(fake_disc_logits), name='d3')
d_loss = d_loss1 + (d_loss2 + d_loss3)
g_loss = tl.cost.sigmoid_cross_entropy(fake_disc_logits, tf.ones_like(fake_disc_logits), name='g')

tf.summary.scalar("g_loss", g_loss)
tf.summary.scalar("d_loss", d_loss)

lr = 0.0002
t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_vars)

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
 
    while epoch!=1000:
        dl, gl = [],[]
      
        if epoch % 50 == 0 :
            saver.save(sess, LOGDIR + "/model.ckpt", global_step=epoch)
  
        for i in range(n_batches):
            z_mb, Y_gen, box_generated = sample_generator_input(BATCH_SIZE, Z_DIM)
            data_1, label_1, bbox_1 = next_batch(BATCH_SIZE)
            data_1 = process_oneimg(data_1)
            feed_dict = {Y:label_1,X:data_1,b:bbox_1,z:z_mb,Y_:Y_gen,b_:box_generated}
            _, DLOSS = sess.run([d_optim, d_loss],feed_dict=feed_dict)
            _, GLOSS = sess.run([g_optim, g_loss],feed_dict=feed_dict)

      
        dl.append(DLOSS)
        gl.append(GLOSS)
        print('discriminator_loss / generator_loss => %.2f / %.2f for step %d'%(np.mean(dl), np.mean(gl), epoch))
        z_mb, Y_gen, box_generated = sample_generator_input(BATCH_SIZE, Z_DIM)
        data_1, label_1, bbox_1 = next_batch(BATCH_SIZE)
        data_1 = process_oneimg(data_1)
        feed_dict = {Y:label_1,X:data_1,b:bbox_1,z:z_mb,Y_:Y_gen,b_:box_generated}
        fake_image = sess.run(sample,feed_dict=feed_dict)
        plt.imsave('new_result/'+str(epoch)+'.jpg',transform(fake_image[0]))
        epoch+=1

  

