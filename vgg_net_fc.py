"""VGG16 network architecture
Author: Feng Liu, liufeng@stanford.edu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def vgg_net(X, n_classes):
    """ VGG16 network architecture

    Args: 
        X: numpy matrix of np.uint8, [batch_size, height, width, channel]
        n_classes: number of classes
    """
    # convert X to tf.float32
    X = tf.cast(X, tf.float32)

    # resize images to 224x224x3
    X =  tf.image.resize_image_with_crop_or_pad(X, 224, 224)
    
    # restore parameters
    f = np.load('vgg16_weights.npz')

    # setup variables
    Wconv1 = tf.get_variable("Wconv1",# shape = [3, 3, 3, 64],
                             initializer = tf.constant(f['conv1_1_W']), 
                             trainable = False)
    bconv1 = tf.get_variable("bconv1",# shape=[64],
                             initializer = tf.constant(f['conv1_1_b']), 
                             trainable = False)
    Wconv2 = tf.get_variable("Wconv2",# shape=[3, 3, 64, 64],
                             initializer = tf.constant(f['conv1_2_W']), 
                             trainable = False)
    bconv2 = tf.get_variable("bconv2",# shape=[64],
                             initializer = tf.constant(f['conv1_2_b']), 
                             trainable = False)

    Wconv3 = tf.get_variable("Wconv3",# shape=[3, 3, 64, 128],
                             initializer = tf.constant(f['conv2_1_W']), 
                             trainable = False)
    bconv3 = tf.get_variable("bconv3",# shape=[128],
                             initializer = tf.constant(f['conv2_1_b']), 
                             trainable = False)
    Wconv4 = tf.get_variable("Wconv4",# shape=[3, 3, 128, 128],
                             initializer = tf.constant(f['conv2_2_W']), 
                             trainable = False)
    bconv4 = tf.get_variable("bconv4",# shape=[128],
                             initializer = tf.constant(f['conv2_2_b']), 
                             trainable = False)

    Wconv5 = tf.get_variable("Wconv5",# shape=[3, 3, 128, 256],
                             initializer = tf.constant(f['conv3_1_W']), 
                             trainable = False)
    bconv5 = tf.get_variable("bconv5",# shape=[256],
                             initializer = tf.constant(f['conv3_1_b']), 
                             trainable = False)
    Wconv6 = tf.get_variable("Wconv6",# shape=[3, 3, 256, 256],
                             initializer = tf.constant(f['conv3_2_W']), 
                             trainable = False)
    bconv6 = tf.get_variable("bconv6",# shape=[256],
                             initializer = tf.constant(f['conv3_2_b']), 
                             trainable = False)
    Wconv7 = tf.get_variable("Wconv7",# shape=[3, 3, 256, 256],
                             initializer = tf.constant(f['conv3_3_W']), 
                             trainable = False)
    bconv7 = tf.get_variable("bconv7",# shape=[256],
                             initializer = tf.constant(f['conv3_3_b']), 
                             trainable = False)

    Wconv8 = tf.get_variable("Wconv8",# shape=[3, 3, 256, 512],
                             initializer = tf.constant(f['conv4_1_W']), 
                             trainable = False)
    bconv8 = tf.get_variable("bconv8",# shape=[512],
                             initializer = tf.constant(f['conv4_1_b']), 
                             trainable = False)
    Wconv9 = tf.get_variable("Wconv9",# shape=[3, 3, 512, 512],
                             initializer = tf.constant(f['conv4_2_W']), 
                             trainable = False)
    bconv9 = tf.get_variable("bconv9",# shape=[512],
                             initializer = tf.constant(f['conv4_2_b']), 
                             trainable = False)
    Wconv10 = tf.get_variable("Wconv10",# shape=[3, 3, 512, 512],
                             initializer = tf.constant(f['conv4_3_W']), 
                             trainable = False)
    bconv10 = tf.get_variable("bconv10",# shape=[512],
                             initializer = tf.constant(f['conv4_3_b']), 
                             trainable = False)

    Wconv11 = tf.get_variable("Wconv11",# shape=[3, 3, 256, 512],
                             initializer = tf.constant(f['conv5_1_W']), 
                             trainable = False)
    bconv11 = tf.get_variable("bconv11",# shape=[512],
                             initializer = tf.constant(f['conv5_1_b']), 
                             trainable = False)
    Wconv12 = tf.get_variable("Wconv12",# shape=[3, 3, 512, 512],
                             initializer = tf.constant(f['conv5_2_W']), 
                             trainable = False)
    bconv12 = tf.get_variable("bconv12",# shape=[512],
                             initializer = tf.constant(f['conv5_2_b']), 
                             trainable = False)
    Wconv13 = tf.get_variable("Wconv13",# shape=[3, 3, 512, 512],
                             initializer = tf.constant(f['conv5_3_W']), 
                             trainable = False)
    bconv13 = tf.get_variable("bconv13",# shape=[512],
                             initializer = tf.constant(f['conv5_3_b']), 
                             trainable = False)

    W1 = tf.get_variable("W1",# shape=[25088, 4096],
                        initializer = tf.constant(f['fc6_W']), 
                        trainable = True) # 7x7x512
    b1 = tf.get_variable("b1",# shape=[4096],
                        initializer = tf.constant(f['fc6_b']), 
                        trainable = True)
    W2 = tf.get_variable("W2",# shape=[4096, 4096],
                        initializer = tf.constant(f['fc7_W']), 
                        trainable = True)
    b2 = tf.get_variable("b2",# shape=[4096],
                        initializer = tf.constant(f['fc7_b']), 
                        trainable = True)
    W3 = tf.get_variable("W3", shape=[4096, n_classes],
                        # initializer = tf.constant(f['fc8_W']), 
                        trainable = True)
    b3 = tf.get_variable("b3", shape=[n_classes],
                        # initializer = tf.constant(f['fc8_b']), 
                        trainable = True)

    # define our graph
    h1 = tf.nn.relu(tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='SAME') + bconv1)
    h2 = tf.nn.relu(tf.nn.conv2d(h1, Wconv2, strides=[1,1,1,1], padding='SAME') + bconv2)
    p2 = tf.nn.max_pool(h2, [1,2,2,1], strides=[1,2,2,1], padding='SAME')

    h3 = tf.nn.relu(tf.nn.conv2d(p2, Wconv3, strides=[1,1,1,1], padding='SAME') + bconv3)
    h4 = tf.nn.relu(tf.nn.conv2d(h3, Wconv4, strides=[1,1,1,1], padding='SAME') + bconv4)
    p4 = tf.nn.max_pool(h4, [1,2,2,1], strides=[1,2,2,1], padding='SAME')

    h5 = tf.nn.relu(tf.nn.conv2d(p4, Wconv5, strides=[1,1,1,1], padding='SAME') + bconv5)
    h6 = tf.nn.relu(tf.nn.conv2d(h5, Wconv6, strides=[1,1,1,1], padding='SAME') + bconv6)
    h7 = tf.nn.relu(tf.nn.conv2d(h6, Wconv7, strides=[1,1,1,1], padding='SAME') + bconv7)
    p7 = tf.nn.max_pool(h7, [1,2,2,1], strides=[1,2,2,1], padding='SAME')

    h8 = tf.nn.relu(tf.nn.conv2d(p7, Wconv8, strides=[1,1,1,1], padding='SAME') + bconv8)
    h9 = tf.nn.relu(tf.nn.conv2d(h8, Wconv9, strides=[1,1,1,1], padding='SAME') + bconv9)
    h10 = tf.nn.relu(tf.nn.conv2d(h9, Wconv10, strides=[1,1,1,1], padding='SAME') + bconv10)
    p10 = tf.nn.max_pool(h10, [1,2,2,1], strides=[1,2,2,1], padding='SAME')

    h11 = tf.nn.relu(tf.nn.conv2d(p10, Wconv11, strides=[1,1,1,1], padding='SAME') + bconv11)
    h12 = tf.nn.relu(tf.nn.conv2d(h11, Wconv12, strides=[1,1,1,1], padding='SAME') + bconv12)
    h13 = tf.nn.relu(tf.nn.conv2d(h12, Wconv13, strides=[1,1,1,1], padding='SAME') + bconv13)
    p13 = tf.nn.max_pool(h13, [1,2,2,1], strides=[1,2,2,1], padding='SAME')


    fc1 = tf.nn.relu(tf.matmul(tf.reshape(p13, [-1, 25088]), W1) + b1)
    # dp1 = tf.nn.dropout(fc1, keep_prob = 1)
    fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
    y_out = tf.matmul(fc2, W3) + b3
    return y_out
