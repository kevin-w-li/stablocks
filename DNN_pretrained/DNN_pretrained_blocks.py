################################################################################
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################
import matplotlib
matplotlib.use('Agg')
from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import h5py
import tensorflow as tf
from utils import *
import shutil

np.random.seed(0)
tf.set_random_seed(0)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('image_dim', 227, 'first dimension of the image; we are assuming a square image')
flags.DEFINE_integer('color_channel', 3, 'number of color channels')
flags.DEFINE_integer('num_gridlines', 100, 'number of grid lines')
flags.DEFINE_integer('num_minibatches', 100, 'number of minibatches')
flags.DEFINE_string('exp_name', '100000_5_5', 'some informative name for the experiment')

################################################################################


net_data = load("bvlc_alexnet.npy").item()


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])



def Alexnet(input_shape=[None, FLAGS.image_dim, FLAGS.image_dim, FLAGS.color_channel],
        output_shape=[None, FLAGS.num_gridlines]):

    x = tf.placeholder(tf.float32, input_shape)
    y = tf.placeholder(tf.float32, output_shape)

    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11;
    k_w = 11;
    c_o = 96;
    s_h = 4;
    s_w = 4
    conv1W = tf.constant(net_data["conv1"][0])
    conv1b = tf.constant(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5;
    k_w = 5;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv2W = tf.constant(net_data["conv2"][0])
    conv2b = tf.constant(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 1
    conv3W = tf.constant(net_data["conv3"][0])
    conv3b = tf.constant(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3;
    k_w = 3;
    c_o = 384;
    s_h = 1;
    s_w = 1;
    group = 2
    conv4W = tf.constant(net_data["conv4"][0])
    conv4b = tf.constant(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3;
    k_w = 3;
    c_o = 256;
    s_h = 1;
    s_w = 1;
    group = 2
    conv5W = tf.constant(net_data["conv5"][0])
    conv5b = tf.constant(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3;
    k_w = 3;
    s_h = 2;
    s_w = 2;
    padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # fc6
    # fc(4096, name='fc6')
    fc6W = tf.constant(net_data["fc6"][0])
    fc6b = tf.constant(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    # fc7
    # fc(4096, name='fc7')
    fc7W = tf.constant(net_data["fc7"][0])
    fc7b = tf.constant(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    W_regression = weight_variable([4096, FLAGS.num_gridlines])
    b_regression = bias_variable([FLAGS.num_gridlines])
    h_regression = tf.nn.sigmoid(tf.matmul(fc7, W_regression) + b_regression)

    temp_dist = tf.sub(y, h_regression)
    SE = tf.nn.l2_loss(temp_dist)
    MSE = tf.reduce_mean(SE, name='mse')
    return {'cost': MSE, 'y_output': y, 'x_input': x, 'y': h_regression}



def test_DNN_pretrained():
    # hdf_file = h5py.File('../data/dataset_10000_5_10.hdf5', 'r')
    # images = hdf_file.get('data')
    # labels = hdf_file.get('label')
    # num_images = images.shape[0]
    # images = np.reshape(images, (FLAGS.num_minibatches, num_images / float(FLAGS.num_minibatches), FLAGS.image_dim, FLAGS.image_dim, FLAGS.color_channel))
    # labels = np.reshape(labels, (FLAGS.num_minibatches, num_images / float(FLAGS.num_minibatches), FLAGS.num_gridlines))
    # x_train = images[:9000]
    # y_train = labels[:9000]
    # x_test = images[1000:]
    # y_test = labels[1000:]
    hdf_file = h5py.File('../data/dataset_1000_5.hdf5', 'r')
    images = hdf_file.get('data')
    labels = hdf_file.get('label')
    num_images = images.shape[0]
    images = np.reshape(images, (
    FLAGS.num_minibatches, num_images / float(FLAGS.num_minibatches), FLAGS.image_dim, FLAGS.image_dim,
    FLAGS.color_channel))
    labels = np.reshape(labels, (FLAGS.num_minibatches, num_images / float(FLAGS.num_minibatches), FLAGS.num_gridlines))
    x_train = images[:90]
    y_train = labels[:90]
    x_test = images[90:]
    y_test = labels[90:]

    cnn = Alexnet()
    n_epochs = 400
    learning_rate = 0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-07).minimize(cnn['cost'])

    try:
        shutil.rmtree('logs/' + FLAGS.exp_name)
    except:
        pass
    if not os.path.exists('logs/' + FLAGS.exp_name):
        os.makedirs('logs/' + FLAGS.exp_name)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for epoch_i in range(n_epochs):
        print('--- Epoch', epoch_i)
        train_cost = 0
        for batch_i in np.random.permutation(range(y_train.shape[0])):
            tic = time.time()
            print batch_i
            batch_y_output = y_train[batch_i, :, :]
            batch_X_input = x_train[batch_i, :, :, :, :]
            temp = sess.run([cnn['cost'], optimizer],
                            feed_dict={cnn['y_output']: batch_y_output, cnn['x_input']: batch_X_input})[0]
            if batch_i == 1:
                saver = tf.train.Saver()
                saver.save(sess, 'logs/' + FLAGS.exp_name + '/', global_step=0)


            # print 'Minibatch cost: ', temp
            train_cost += temp
            toc = time.time()
            # print 'time per minibatch: ', toc - tic
        print('Train cost:', train_cost / (y_train.shape[0]))
        text_file = open("logs/" + FLAGS.exp_name + "/train_costs.txt", "a")
        text_file.write(str(train_cost / (y_train.shape[0])))
        text_file.write('\n')
        text_file.close()

        #################################### Testing ########################################
        valid_cost = 0
        for batch_i in range(y_test.shape[0]):
            batch_y_output = y_test[batch_i, :, :]
            batch_X_input = x_test[batch_i, :, :, :, :]
            valid_cost += sess.run([cnn['cost']], feed_dict={cnn['y_output']: batch_y_output,
                                                            cnn['x_input']: batch_X_input})[0]
            true_mat = batch_y_output
            pred_mat = sess.run([cnn['y']][0], feed_dict={cnn['y_output']: batch_y_output,
                                                         cnn['x_input']: batch_X_input})
        fig0, axes0 = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
        axes0[0][0].imshow(true_mat, aspect='auto', interpolation="nearest", vmin=0, vmax=1)
        #axes0[0][0].set_xticks(range(11))
        axes0[0][0].set_title('True probs')
        axes0[0][1].imshow(pred_mat, aspect='auto', interpolation="nearest", vmin=0, vmax=1)
        #axes0[0][1].set_xticks(range(11))
        axes0[0][1].set_title('Pred probs')
#        plt.colorbar()
        plt.savefig('logs/' + FLAGS.exp_name + '/prob_mat.png', bbox_inches='tight')
        print('Validation cost:', valid_cost / (y_test.shape[0]))
        text_file = open("logs/" + FLAGS.exp_name + "/test_costs.txt", "a")
        text_file.write(str(valid_cost / (y_test.shape[0])))
        text_file.write('\n')
        text_file.close()



if __name__ == '__main__':
    test_DNN_pretrained()
