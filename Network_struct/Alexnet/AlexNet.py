from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import argparse

from datetime import datetime
import math
import sys
import time
from six.moves import xrange
import _pickle as pickle
import os


FLAGS = None


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def Kernel(shape):
    Weight = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name="conv_kernel")
    return Weight


def biases(shape):
    biase= tf.Variable(tf.constant(shape, dtype=tf.float32), trainable=True, name="biases")
    return biase


def AlexNet(train_x, train_y, test_x, test_y):
    """Buiild Alexnet model"""

    images = tf.placeholder(dtype=tf.float32, shape=[None, 150528])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    images_x = tf.reshape(images, [-1, 224, 224, 3])
    #Convelution Layer 1
    with tf.name_scope('conv1') as scope:
        kernel = Kernel([11, 11, 3, 96])
        bias = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=True, name='biases')
        conv1 = tf.nn.conv2d(images_x, kernel, strides=[1, 4, 4, 1], padding='SAME', name=scope)
        activation = tf.nn.relu(tf.nn.bias_add(conv1, bias), name='activation1')

    pool1 = tf.nn.max_pool(activation, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)

    # Convelution Layer 2
    with tf.name_scope('conv2') as scope:
        kernel = Kernel([5, 5, 96, 256])
        bias = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        conv2 = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME', name=scope)
        activation = tf.nn.relu(tf.nn.bias_add(conv2, bias), name='activation2')

    pool2 = tf.nn.max_pool(activation, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activations(pool2)

    # Convelution Layer 3
    with tf.name_scope('conv3') as scope:
        kernel = Kernel([3, 3, 256, 384])
        bias = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        conv3 = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME', name=scope)
        activation = tf.nn.relu(tf.nn.bias_add(conv3, bias), name='activation3')

    # Convelution Layer 4
    with tf.name_scope('conv4') as scope:
        kernel = Kernel([3, 3, 384, 384])
        bias = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        conv4 = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME', name=scope)
        activation = tf.nn.relu(tf.nn.bias_add(conv4, bias), name='activation4')

    # Convelution Layer 5
    with tf.name_scope('conv5') as scope:
        kernel = Kernel([3, 3, 384, 256])
        bias = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        conv5 = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1], padding='SAME', name=scope)
        activation = tf.nn.relu(tf.nn.bias_add(conv5, bias), name='activation5')

    pool5 = tf.nn.max_pool(activation, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)

    flatten = tf.reshape(pool5, [-1, 6*6*256])
    # Fully connecting Layer1
    with tf.name_scope('Fully_connecting_Layer1') as scope:
        weights = Kernel([9216, 4096])
        bias = biases([4096])
        FCL1 = tf.matmul(flatten, weights, name=scope) + bias
        activation1 = tf.nn.relu(FCL1)

    # Fully connecting Layer2
    with tf.name_scope('Fully_connecting_Layer2') as scope:
        weights = Kernel([4096, 4096])
        bias = biases([4096])
        FCL2 = tf.matmul(activation1, weights, name=scope) + bias
        activation2 = tf.nn.relu(FCL2)

    # Fully connecting Layer3
    with tf.name_scope('Fully_connecting_Layer3') as scope:
        weights = Kernel([4096, 10])
        bias = biases([10])
        FCL3 = tf.matmul(activation2, weights, name=scope) + bias

    prediction = tf.nn.sigmoid(FCL3)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=prediction))
    train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

    prediction_2 = tf.nn.softmax(prediction)
    correct_prediction = tf.equal(tf.argmax(prediction_2, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(20):
            for batch in range(20):
                sess.run(train_step, feed_dict={images: train_x[batch], labels: train_y[batch]})
                acc = sess.run(accuracy, feed_dict={images: test_x, labels: test_y})
            print("Accuracy:", str(acc))
    return prediction


def Cifar10_Load(file):

    with open(file, 'rb') as f:
        cifar10 = pickle.load(f, encoding='iso-8859-1')
        data = cifar10['data']
        label = cifar10['labels']
        label = np.array(label)
        label = to_categorical(label)
        # print("labelsssss: ", label)
    return data, label


def main():

    batch_size = 20
    n_batch = 250
    keep_prob = tf.placeholder(tf.float32)

    X_train, Y_train = Cifar10_Load('data_train_1')
    X_test, Y_test = Cifar10_Load('data_test_1')
    state = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(Y_train)

    batch_x = []
    batch_y = []
    # for batch in range(n_batch):
    #     batch_x.append([])
    #     batch_y.append([])


    for i in range(n_batch):
        batch_x.append(X_train[i*batch_size:(i+1)*batch_size-1])
        print("[",i*batch_size,':',(i+1)*batch_size-1,']')
        print([X_train[i*batch_size:(i+1)*batch_size-1]])
        batch_y.append(Y_train[i*batch_size:(i+1)*batch_size-1])
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    print("batch_x：", batch_x.shape)
    print("batch_y：", batch_y.shape)
    print('batch_20:', batch_y[19])

    AlexNet(batch_x, batch_y, X_test, Y_test)

main()










