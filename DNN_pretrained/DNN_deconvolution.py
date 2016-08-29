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
flags.DEFINE_integer('num_images', 500, 'number of minibatches')
flags.DEFINE_string('exp_name', 'deconv_1000_5', 'some informative name for the experiment')

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

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


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

    W_regression = weight_variable([4096, 64])
    b_regression = bias_variable([64])
    h_regression = tf.nn.relu(tf.matmul(fc7, W_regression) + b_regression)
    #
    # W2_regression = weight_variable([64 * 64, 227 * 227])
    # b2_regression = bias_variable([227 * 227])
    # h2_regression = tf.nn.sigmoid(tf.matmul(h_regression, W2_regression) + b2_regression)

    #
    # temp_dist = tf.sub(y, h_regression)
    # SE = tf.nn.l2_loss(temp_dist)
    # MSE = tf.reduce_mean(SE, name='mse')
    # return {'cost': MSE, 'y_output': y, 'x_input': x, 'y': h_regression}

    # #####Deconvolution to the heatmap

    deconv000_weight = weight_variable([3, 3, 1, 1])
    convt_000 = tf.nn.relu(
        tf.nn.conv2d_transpose(tf.reshape(h_regression, [-1, 8, 8, 1]),
                               filter=deconv000_weight,
                               output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 16, 16, 1],
                               strides=[1, 2, 2, 1]))

    deconv00_weight = weight_variable([3, 3, 1, 1])
    convt_00 = tf.nn.relu(
        tf.nn.conv2d_transpose(tf.reshape(convt_000, [-1, 16, 16, 1]),
                               filter=deconv00_weight,
                               output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 32, 32, 1],
                               strides=[1, 1, 1, 1]))

    deconv0_weight = weight_variable([5, 5, 1, 1])
    convt_0 = tf.nn.relu(
        tf.nn.conv2d_transpose(tf.reshape(convt_00, [-1, 32, 32, 1]),
                               filter=deconv0_weight,
                               output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 64, 64, 1],
                               strides=[1, 2, 2, 1]))

    deconv1_weight = weight_variable([5, 5, 1, 1])
    convt_1 = tf.nn.relu(
        tf.nn.conv2d_transpose(tf.reshape(convt_0, [-1, 64, 64, 1]),
                               filter=deconv1_weight,
                               output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 128, 128, 1],
                               strides=[1, 2, 2, 1]))
    convt_1_drop = tf.nn.dropout(convt_1, 0.4)
    deconv2_weight = weight_variable([7, 7, 1, 1])
    convt_2 = tf.nn.sigmoid(
        tf.nn.conv2d_transpose(convt_1_drop,
                               filter=deconv2_weight,
                               output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), FLAGS.image_dim + 29, FLAGS.image_dim + 29, 1],
                               strides=[1, 2, 2, 1]))

    target_resized =  tf.slice(x, [0, 0, 0, 0], [FLAGS.num_images / (FLAGS.num_minibatches), FLAGS.image_dim - 1, FLAGS.image_dim - 1, 1])
    target_resized = tf.pad(target_resized, [[0, 0], [15, 15], [15, 15], [0,0]], "CONSTANT")
    target_grayscaled = (tf.image.rgb_to_grayscale(target_resized) / 255.)
    temp_dist = tf.sub(target_grayscaled, convt_2)
    SE = tf.nn.l2_loss(temp_dist)
    MSE = tf.reduce_mean(SE, name='mse')
    return {'cost': MSE, 'y_output': y, 'x_input': x, 'y': convt_2,
            'grayscale': target_grayscaled}


def test_DNN_pretrained():
    hdf_file = h5py.File('../data/debug_dataset_5_3_10.hdf5', 'r')
    images = hdf_file.get('data')
    labels = hdf_file.get('label')
    num_images = images.shape[0]
    images = np.reshape(images, (FLAGS.num_minibatches, num_images / float(FLAGS.num_minibatches), FLAGS.image_dim, FLAGS.image_dim, FLAGS.color_channel))
    labels = np.reshape(labels, (FLAGS.num_minibatches, num_images / float(FLAGS.num_minibatches), FLAGS.num_gridlines))
    x_train = images[:90]
    y_train = labels[:90]
    x_test = images[90:]
    y_test = labels[90:]
#check why the blue area doesn't get blue '
    cnn = Alexnet()
    n_epochs = 400
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.4, beta2=0.7, epsilon=1e-5).minimize(cnn['cost'])
    #optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cnn['cost'])
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
            # if batch_i == 1:
            #     saver = tf.train.Saver()
            #     saver.save(sess, 'logs/' + FLAGS.exp_name + '/', global_step=0)


            print 'Minibatch cost: ', temp
            train_cost += temp
            toc = time.time()
            print 'time per minibatch: ', toc - tic
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
                                                         cnn['x_input']: batch_X_input})[0]
            target = sess.run(cnn['grayscale'], feed_dict = {cnn['y_output']: batch_y_output, cnn['x_input']: batch_X_input})[0]

        fig0, axes0 = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
        axes0[0][0].imshow(target.squeeze(), aspect='auto', interpolation="nearest", vmin=0, vmax=1)
        #axes0[0][0].set_xticks(range(11))
        axes0[0][0].set_title('True probs')
        axes0[0][1].imshow(pred_mat.squeeze(), aspect='auto', interpolation="nearest", vmin=0, vmax=1)
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






#######################################
#







############################################# GARBAGE ##############################

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
    # import matplotlib
    #
    # matplotlib.use('Agg')
    # from numpy import *
    # import os
    # from pylab import *
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import matplotlib.cbook as cbook
    # import time
    # from scipy.misc import imread
    # from scipy.misc import imresize
    # import matplotlib.image as mpimg
    # from scipy.ndimage import filters
    # import urllib
    # from numpy import random
    # import h5py
    # import tensorflow as tf
    # from utils import *
    # import shutil
    #
    # np.random.seed(0)
    # tf.set_random_seed(0)
    #
    # flags = tf.app.flags
    # FLAGS = flags.FLAGS
    #
    # flags.DEFINE_integer('image_dim', 227, 'first dimension of the image; we are assuming a square image')
    # flags.DEFINE_integer('color_channel', 3, 'number of color channels')
    # flags.DEFINE_integer('num_gridlines', 100, 'number of grid lines')
    # flags.DEFINE_integer('num_minibatches', 100, 'number of minibatches')
    # flags.DEFINE_integer('num_images', 500, 'number of minibatches')
    # flags.DEFINE_string('exp_name', 'deconv_1000_5', 'some informative name for the experiment')
    #
    # ################################################################################
    #
    #
    # net_data = load("bvlc_alexnet.npy").item()
    #
    #
    # def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    #     '''From https://github.com/ethereon/caffe-tensorflow
    #     '''
    #     c_i = input.get_shape()[-1]
    #     assert c_i % group == 0
    #     assert c_o % group == 0
    #     convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    #
    #     if group == 1:
    #         conv = convolve(input, kernel)
    #     else:
    #         input_groups = tf.split(3, group, input)
    #         kernel_groups = tf.split(3, group, kernel)
    #         output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    #         conv = tf.concat(3, output_groups)
    #     return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])
    #
    #
    # def conv2d(x, W):
    #     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #
    #
    # def Alexnet(input_shape=[None, FLAGS.image_dim, FLAGS.image_dim, FLAGS.color_channel],
    #             output_shape=[None, FLAGS.num_gridlines]):
    #
    #     x = tf.placeholder(tf.float32, input_shape)
    #     y = tf.placeholder(tf.float32, output_shape)
    #
    #     # conv1
    #     # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    #     k_h = 11;
    #     k_w = 11;
    #     c_o = 96;
    #     s_h = 4;
    #     s_w = 4
    #     conv1W = tf.constant(net_data["conv1"][0])
    #     conv1b = tf.constant(net_data["conv1"][1])
    #     conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    #     conv1 = tf.nn.relu(conv1_in)
    #
    #     # lrn1
    #     # lrn(2, 2e-05, 0.75, name='norm1')
    #     radius = 2;
    #     alpha = 2e-05;
    #     beta = 0.75;
    #     bias = 1.0
    #     lrn1 = tf.nn.local_response_normalization(conv1,
    #                                               depth_radius=radius,
    #                                               alpha=alpha,
    #                                               beta=beta,
    #                                               bias=bias)
    #
    #     # maxpool1
    #     # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    #     k_h = 3;
    #     k_w = 3;
    #     s_h = 2;
    #     s_w = 2;
    #     padding = 'VALID'
    #     maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    #
    #     # conv2
    #     # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    #     k_h = 5;
    #     k_w = 5;
    #     c_o = 256;
    #     s_h = 1;
    #     s_w = 1;
    #     group = 2
    #     conv2W = tf.constant(net_data["conv2"][0])
    #     conv2b = tf.constant(net_data["conv2"][1])
    #     conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #     conv2 = tf.nn.relu(conv2_in)
    #
    #     # lrn2
    #     # lrn(2, 2e-05, 0.75, name='norm2')
    #     radius = 2;
    #     alpha = 2e-05;
    #     beta = 0.75;
    #     bias = 1.0
    #     lrn2 = tf.nn.local_response_normalization(conv2,
    #                                               depth_radius=radius,
    #                                               alpha=alpha,
    #                                               beta=beta,
    #                                               bias=bias)
    #
    #     # maxpool2
    #     # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    #     k_h = 3;
    #     k_w = 3;
    #     s_h = 2;
    #     s_w = 2;
    #     padding = 'VALID'
    #     maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    #
    #     # conv3
    #     # conv(3, 3, 384, 1, 1, name='conv3')
    #     k_h = 3;
    #     k_w = 3;
    #     c_o = 384;
    #     s_h = 1;
    #     s_w = 1;
    #     group = 1
    #     conv3W = tf.constant(net_data["conv3"][0])
    #     conv3b = tf.constant(net_data["conv3"][1])
    #     conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #     conv3 = tf.nn.relu(conv3_in)
    #
    #     # conv4
    #     # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    #     k_h = 3;
    #     k_w = 3;
    #     c_o = 384;
    #     s_h = 1;
    #     s_w = 1;
    #     group = 2
    #     conv4W = tf.constant(net_data["conv4"][0])
    #     conv4b = tf.constant(net_data["conv4"][1])
    #     conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #     conv4 = tf.nn.relu(conv4_in)
    #
    #     # conv5
    #     # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    #     k_h = 3;
    #     k_w = 3;
    #     c_o = 256;
    #     s_h = 1;
    #     s_w = 1;
    #     group = 2
    #     conv5W = tf.constant(net_data["conv5"][0])
    #     conv5b = tf.constant(net_data["conv5"][1])
    #     conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #     conv5 = tf.nn.relu(conv5_in)
    #
    #     # maxpool5
    #     # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    #     k_h = 3;
    #     k_w = 3;
    #     s_h = 2;
    #     s_w = 2;
    #     padding = 'VALID'
    #     maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    #
    #     # fc6
    #     # fc(4096, name='fc6')
    #     fc6W = tf.constant(net_data["fc6"][0])
    #     fc6b = tf.constant(net_data["fc6"][1])
    #     fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
    #
    #     # fc7
    #     # fc(4096, name='fc7')
    #     fc7W = tf.constant(net_data["fc7"][0])
    #     fc7b = tf.constant(net_data["fc7"][1])
    #     fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
    #
    #     W_regression = weight_variable([4096, 64 * 64])
    #     b_regression = bias_variable([64 * 64])
    #     h_regression = tf.nn.relu(tf.matmul(fc7, W_regression) + b_regression)
    #
    #     W2_regression = weight_variable([64 * 64, 227 * 227])
    #     b2_regression = bias_variable([227 * 227])
    #     h2_regression = tf.nn.sigmoid(tf.matmul(h_regression, W2_regression) + b2_regression)
    #
    #     #
    #     # temp_dist = tf.sub(y, h_regression)
    #     # SE = tf.nn.l2_loss(temp_dist)
    #     # MSE = tf.reduce_mean(SE, name='mse')
    #     # return {'cost': MSE, 'y_output': y, 'x_input': x, 'y': h_regression}
    #
    #     # #####Deconvolution to the heatmap
    #
    #     # deconv000_weight = weight_variable([3, 3, 1, 1])
    #     # convt_000 = tf.nn.relu(
    #     #     tf.nn.conv2d_transpose(tf.reshape(h_regression, [-1, 8, 8, 1]),
    #     #                            filter=deconv000_weight,
    #     #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 16, 16, 1],
    #     #                            strides=[1, 2, 2, 1]))
    #     #
    #     # deconv00_weight = weight_variable([7, 7, 1, 1])
    #     # convt_00 = tf.nn.relu(
    #     #     tf.nn.conv2d_transpose(tf.reshape(h_regression, [-1, 16, 16, 1]),
    #     #                            filter=deconv00_weight,
    #     #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 32, 32, 1],
    #     #                            strides=[1, 2, 2, 1]))
    #
    #     # deconv0_weight = weight_variable([5, 5, 1, 1])
    #     # convt_0 = tf.nn.relu(
    #     #     tf.nn.conv2d_transpose(tf.reshape(h_regression, [-1, 32, 32, 1]),
    #     #                            filter=deconv0_weight,
    #     #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 64, 64, 1],
    #     #                            strides=[1, 2, 2, 1]))
    #
    #     # deconv1_weight = weight_variable([5, 5, 1, 1])
    #     # convt_1 = tf.nn.relu(
    #     #     tf.nn.conv2d_transpose(tf.reshape(h_regression, [-1, 64, 64, 1]),
    #     #                            filter=deconv1_weight,
    #     #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 128, 128, 1],
    #     #                            strides=[1, 2, 2, 1]))
    #     #
    #     # deconv2_weight = weight_variable([7, 7, 1, 1])
    #     # convt_2 = tf.nn.sigmoid(
    #     #     tf.nn.conv2d_transpose(convt_1,
    #     #                            filter=deconv2_weight,
    #     #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), FLAGS.image_dim + 29, FLAGS.image_dim + 29, 1],
    #     #                            strides=[1, 2, 2, 1]))
    #
    #     target_resized = x  # tf.slice(x, [0, 0, 0, 0], [FLAGS.num_images / (FLAGS.num_minibatches), FLAGS.image_dim - 1, FLAGS.image_dim - 1, 1])
    #     # target_resized = tf.pad(target_resized, [[0, 0], [15, 15], [15, 15], [0,0]], "CONSTANT")
    #     target_grayscaled = (tf.image.rgb_to_grayscale(target_resized) / 255.)
    #     temp_dist = tf.sub(target_grayscaled, tf.reshape(h2_regression, [-1, 227, 227, 1]))
    #     SE = tf.nn.l2_loss(temp_dist)
    #     MSE = tf.reduce_mean(SE, name='mse')
    #     return {'cost': MSE, 'y_output': y, 'x_input': x, 'y': tf.reshape(h2_regression, [-1, 227, 227, 1]),
    #             'grayscale': target_grayscaled}
    #
    #
    # def test_DNN_pretrained():
    #     hdf_file = h5py.File('../data/debug_dataset_5_3_10.hdf5', 'r')
    #     images = hdf_file.get('data')
    #     labels = hdf_file.get('label')
    #     num_images = images.shape[0]
    #     images = np.reshape(images, (
    #     FLAGS.num_minibatches, num_images / float(FLAGS.num_minibatches), FLAGS.image_dim, FLAGS.image_dim,
    #     FLAGS.color_channel))
    #     labels = np.reshape(labels,
    #                         (FLAGS.num_minibatches, num_images / float(FLAGS.num_minibatches), FLAGS.num_gridlines))
    #     x_train = images[:90]
    #     y_train = labels[:90]
    #     x_test = images[90:]
    #     y_test = labels[90:]
    #     # check why the blue area doesn't get blue '
    #     cnn = Alexnet()
    #     n_epochs = 400
    #     learning_rate = 0.001
    #     # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.4, beta2=0.7, epsilon=1e-5).minimize(cnn['cost'])
    #     optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cnn['cost'])
    #     try:
    #         shutil.rmtree('logs/' + FLAGS.exp_name)
    #     except:
    #         pass
    #     if not os.path.exists('logs/' + FLAGS.exp_name):
    #         os.makedirs('logs/' + FLAGS.exp_name)
    #
    #     init = tf.initialize_all_variables()
    #     sess = tf.Session()
    #     sess.run(init)
    #     for epoch_i in range(n_epochs):
    #         print('--- Epoch', epoch_i)
    #         train_cost = 0
    #         for batch_i in np.random.permutation(range(y_train.shape[0])):
    #             tic = time.time()
    #             print batch_i
    #             batch_y_output = y_train[batch_i, :, :]
    #             batch_X_input = x_train[batch_i, :, :, :, :]
    #             temp = sess.run([cnn['cost'], optimizer],
    #                             feed_dict={cnn['y_output']: batch_y_output, cnn['x_input']: batch_X_input})[0]
    #             # if batch_i == 1:
    #             #     saver = tf.train.Saver()
    #             #     saver.save(sess, 'logs/' + FLAGS.exp_name + '/', global_step=0)
    #
    #
    #             print 'Minibatch cost: ', temp
    #             train_cost += temp
    #             toc = time.time()
    #             print 'time per minibatch: ', toc - tic
    #         print('Train cost:', train_cost / (y_train.shape[0]))
    #         text_file = open("logs/" + FLAGS.exp_name + "/train_costs.txt", "a")
    #         text_file.write(str(train_cost / (y_train.shape[0])))
    #         text_file.write('\n')
    #         text_file.close()
    #
    #         #################################### Testing ########################################
    #         valid_cost = 0
    #         for batch_i in range(y_test.shape[0]):
    #             batch_y_output = y_test[batch_i, :, :]
    #             batch_X_input = x_test[batch_i, :, :, :, :]
    #             valid_cost += sess.run([cnn['cost']], feed_dict={cnn['y_output']: batch_y_output,
    #                                                              cnn['x_input']: batch_X_input})[0]
    #             true_mat = batch_y_output
    #             pred_mat = sess.run([cnn['y']][0], feed_dict={cnn['y_output']: batch_y_output,
    #                                                           cnn['x_input']: batch_X_input})[0]
    #             target = \
    #             sess.run(cnn['grayscale'], feed_dict={cnn['y_output']: batch_y_output, cnn['x_input']: batch_X_input})[
    #                 0]
    #
    #         fig0, axes0 = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))
    #         axes0[0][0].imshow(target.squeeze(), aspect='auto', interpolation="nearest", vmin=0, vmax=1)
    #         # axes0[0][0].set_xticks(range(11))
    #         axes0[0][0].set_title('True probs')
    #         axes0[0][1].imshow(pred_mat.squeeze(), aspect='auto', interpolation="nearest", vmin=0, vmax=1)
    #         # axes0[0][1].set_xticks(range(11))
    #         axes0[0][1].set_title('Pred probs')
    #         #        plt.colorbar()
    #         plt.savefig('logs/' + FLAGS.exp_name + '/prob_mat.png', bbox_inches='tight')
    #         print('Validation cost:', valid_cost / (y_test.shape[0]))
    #         text_file = open("logs/" + FLAGS.exp_name + "/test_costs.txt", "a")
    #         text_file.write(str(valid_cost / (y_test.shape[0])))
    #         text_file.write('\n')
    #         text_file.close()
    #
    #
    # if __name__ == '__main__':
    #     test_DNN_pretrained()

# def Alexnet(input_shape=[None, FLAGS.image_dim, FLAGS.image_dim, FLAGS.color_channel],
#             output_shape=[None, FLAGS.num_gridlines]):
#
#
# x = tf.placeholder(tf.float32, input_shape)
# y = tf.placeholder(tf.float32, output_shape)
#
# # conv1
# # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
# k_h = 11;
# k_w = 11;
# c_o = 96;
# s_h = 4;
# s_w = 4
# conv1W = tf.constant(net_data["conv1"][0])
# conv1b = tf.constant(net_data["conv1"][1])
# conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
# conv1 = tf.nn.relu(conv1_in)
#
# # lrn1
# # lrn(2, 2e-05, 0.75, name='norm1')
# radius = 2;
# alpha = 2e-05;
# beta = 0.75;
# bias = 1.0
# lrn1 = tf.nn.local_response_normalization(conv1,
#                                           depth_radius=radius,
#                                           alpha=alpha,
#                                           beta=beta,
#                                           bias=bias)
#
# # maxpool1
# # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
# k_h = 3;
# k_w = 3;
# s_h = 2;
# s_w = 2;
# padding = 'VALID'
# maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
#
# # conv2
# # conv(5, 5, 256, 1, 1, group=2, name='conv2')
# k_h = 5;
# k_w = 5;
# c_o = 256;
# s_h = 1;
# s_w = 1;
# group = 2
# conv2W = tf.constant(net_data["conv2"][0])
# conv2b = tf.constant(net_data["conv2"][1])
# conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
# conv2 = tf.nn.relu(conv2_in)
#
# # lrn2
# # lrn(2, 2e-05, 0.75, name='norm2')
# radius = 2;
# alpha = 2e-05;
# beta = 0.75;
# bias = 1.0
# lrn2 = tf.nn.local_response_normalization(conv2,
#                                           depth_radius=radius,
#                                           alpha=alpha,
#                                           beta=beta,
#                                           bias=bias)
#
# # maxpool2
# # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
# k_h = 3;
# k_w = 3;
# s_h = 2;
# s_w = 2;
# padding = 'VALID'
# maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
#
# # conv3
# # conv(3, 3, 384, 1, 1, name='conv3')
# k_h = 3;
# k_w = 3;
# c_o = 384;
# s_h = 1;
# s_w = 1;
# group = 1
# conv3W = tf.constant(net_data["conv3"][0])
# conv3b = tf.constant(net_data["conv3"][1])
# conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
# conv3 = tf.nn.relu(conv3_in)
#
# # conv4
# # conv(3, 3, 384, 1, 1, group=2, name='conv4')
# k_h = 3;
# k_w = 3;
# c_o = 384;
# s_h = 1;
# s_w = 1;
# group = 2
# conv4W = tf.constant(net_data["conv4"][0])
# conv4b = tf.constant(net_data["conv4"][1])
# conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
# conv4 = tf.nn.relu(conv4_in)
#
# # conv5
# # conv(3, 3, 256, 1, 1, group=2, name='conv5')
# k_h = 3;
# k_w = 3;
# c_o = 256;
# s_h = 1;
# s_w = 1;
# group = 2
# conv5W = tf.constant(net_data["conv5"][0])
# conv5b = tf.constant(net_data["conv5"][1])
# conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
# conv5 = tf.nn.relu(conv5_in)
#
# # maxpool5
# # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
# k_h = 3;
# k_w = 3;
# s_h = 2;
# s_w = 2;
# padding = 'VALID'
# maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
#
# # fc6
# # fc(4096, name='fc6')
# fc6W = tf.constant(net_data["fc6"][0])
# fc6b = tf.constant(net_data["fc6"][1])
# fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
#
# # fc7
# # fc(4096, name='fc7')
# fc7W = tf.constant(net_data["fc7"][0])
# fc7b = tf.constant(net_data["fc7"][1])
# fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
#
# # W_regression = weight_variable([4096, 64 ])
# # b_regression = bias_variable([64])
# # h_regression = tf.nn.relu(tf.matmul(fc7, W_regression) + b_regression)
# #
# # W_conv4 = weight_variable([1, 1, 64, 5])
# # b_conv4 = bias_variable([5])
# # h_conv4 = tf.nn.relu(conv2d(tf.reshape(h_regression, [-1, 1, 1, 64]), W_conv4) + b_conv4)
#
# batch_size = FLAGS.num_images / (FLAGS.num_minibatches)
# W_conv5 = weight_variable([64, 64, 1, 1])
# b_conv5 = bias_variable([1])
# deconv_shape_conv5 = tf.pack([batch_size, 64, 64, 1])
#
# W_pool3 = weight_variable([2, 2, 1, 1])
# b_pool3 = bias_variable([1])
# deconv_shape_pool3 = tf.pack([batch_size, 128, 128, 1])
#
# W_conv6 = weight_variable([5, 5, 1, 1])
# b_conv6 = bias_variable([1])
# deconv_shape_conv6 = tf.pack([batch_size, 128, 128, 1])
#
# W_pool4 = weight_variable([2, 2, 1, 1])
# b_pool4 = bias_variable([1])
# deconv_shape_pool4 = tf.pack([batch_size, 256, 256, 1])
#
# W_conv7 = weight_variable([5, 5, 1, 1])
# b_conv7 = bias_variable([1])
# deconv_shape_conv7 = tf.pack([batch_size, 256, 256, 1])
#
# # Now the conv2d_transpose part. Hopfuly just looking
# # at the encoder part and decoder part side by side
# # will make it clear how it works.
# h_conv5 = tf.nn.relu(tf.nn.conv2d_transpose(tf.reshape(fc7, [-1, 64, 64, 1]), W_conv5, output_shape=deconv_shape_conv5,
#                                             strides=[1, 1, 1, 1],
#                                             padding='SAME') + b_conv5)
# h_pool3 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv5, W_pool3, output_shape=deconv_shape_pool3, strides=[1, 2, 2, 1],
#                                             padding='SAME') + b_pool3)
# h_conv6 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool3, W_conv6, output_shape=deconv_shape_conv6, strides=[1, 1, 1, 1],
#                                             padding='SAME') + b_conv6)
# h_pool4 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv6, W_pool4, output_shape=deconv_shape_pool4, strides=[1, 2, 2, 1],
#                                             padding='SAME') + b_pool4)
# h_conv7 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool4, W_conv7, output_shape=deconv_shape_conv7, strides=[1, 1, 1, 1],
#                                             padding='SAME') + b_conv7)
# # W2_regression = weight_variable([64 * 64, 227 * 227])
# # b2_regression = bias_variable([227 * 227])
# # h2_regression = tf.nn.sigmoid(tf.matmul(h_regression, W2_regression) + b2_regression)
#
#
# #
# # temp_dist = tf.sub(y, h_regression)
# # SE = tf.nn.l2_loss(temp_dist)
# # MSE = tf.reduce_mean(SE, name='mse')
# # return {'cost': MSE, 'y_output': y, 'x_input': x, 'y': h_regression}
#
# # #####Deconvolution to the heatmap
#
# # deconv000_weight = weight_variable([3, 3, 1, 1])
# # convt_000 = tf.nn.relu(
# #     tf.nn.conv2d_transpose(tf.reshape(h_regression, [-1, 8, 8, 1]),
# #                            filter=deconv000_weight,
# #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 16, 16, 1],
# #                            strides=[1, 2, 2, 1]))
# #
# # deconv00_weight = weight_variable([7, 7, 1, 1])
# # convt_00 = tf.nn.relu(
# #     tf.nn.conv2d_transpose(tf.reshape(h_regression, [-1, 16, 16, 1]),
# #                            filter=deconv00_weight,
# #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 32, 32, 1],
# #                            strides=[1, 2, 2, 1]))
#
# # deconv0_weight = weight_variable([5, 5, 1, 1])
# # convt_0 = tf.nn.relu(
# #     tf.nn.conv2d_transpose(tf.reshape(h_regression, [-1, 32, 32, 1]),
# #                            filter=deconv0_weight,
# #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 64, 64, 1],
# #                            strides=[1, 2, 2, 1]))
#
# # deconv1_weight = weight_variable([5, 5, 5, 1])
# # convt_1 = tf.nn.relu(
# #     tf.nn.conv2d_transpose(tf.reshape(h_regression, [-1, 64, 64, 1]),
# #                            filter=deconv1_weight,
# #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), 256, 256, 5],
# #                            strides=[1, 2, 2, 1], padding='SAME'))
# #
# # deconv2_weight = weight_variable([3, 3, 1, 5])
# # convt_2 = tf.nn.sigmoid(
# #     tf.nn.conv2d_transpose(convt_1,
# #                            filter=deconv2_weight,
# #                            output_shape=[FLAGS.num_images / (FLAGS.num_minibatches), FLAGS.image_dim + 29, FLAGS.image_dim + 29, 1],
# #                            strides=[1, 1, 1, 1]))
#
# target_resized = tf.slice(x, [0, 0, 0, 0],
#                           [FLAGS.num_images / (FLAGS.num_minibatches), FLAGS.image_dim - 1, FLAGS.image_dim - 1, 1])
# target_resized = tf.pad(target_resized, [[0, 0], [15, 15], [15, 15], [0, 0]], "CONSTANT")
# target_grayscaled = (tf.image.rgb_to_grayscale(target_resized) / 255.)
# temp_dist = tf.sub(target_grayscaled, h_conv7)
# SE = tf.nn.l2_loss(temp_dist)
# MSE = tf.reduce_mean(SE, name='mse')
# return {'cost': MSE, 'y_output': y, 'x_input': x, 'y': h_conv5, 'grayscale': target_grayscaled}