from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pk
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import h5py
IMAGE_SIZE = 227

# Global constants describing the Blocks data set.
NUM_GRID_LINES = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10#50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10#10000


def read_blocks(filename_queue, sess=None):
  """Reads and parses examples from Blocks data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class BlocksRecord(object):
    pass
  result = BlocksRecord()


  result.height = 32
  result.width = 32
  result.depth = 1

  #image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  #record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.IdentityReader()
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  # print(sess.run(result.key))
  record_bytes = pk.load(value.eval(session=sess))
  #record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = record_bytes[1] #tf.cast(
      #tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = record_bytes[0] #tf.transpose(depth_major, [1, 2, 0])
  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 1] of type.float32.
    label: 2-D Tensor of type.float32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 1] size.
    labels: Labels. 1D tensor of [batch_size * NUM_GRID_LINES] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        enqueue_many = True,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        enqueue_many=True,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, label_batch #tf.reshape(label_batch, [batch_size])


def inputs(eval_data, data_dir, batch_size, eval_dir = None, sess=None, hdf_file = None, batch_num = None):

  """Construct input for Blocks evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the Blocks data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  #tf.image.rgb_to_grayscale(images)
  if not eval_data:
  #   filenames = os.listdir(data_dir) #[os.path.join(data_dir, 'data_batch.bin')]
     num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
  #   filenames = os.listdir(eval_dir) #[os.path.join(data_dir, 'test_batch.bin')]
     num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  # os.listdir(data_dir)
  # for f in filenames:
  #   if not tf.gfile.Exists(data_dir + f):
  #     raise ValueError('Failed to find file: ' + f)


  images = hdf_file.get('data')
  labels = hdf_file.get('label')
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)
  images = images[batch_num * batch_size: (batch_num + 1) * batch_size]
  labels = labels[batch_num * batch_size: (batch_num + 1) * batch_size]
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(images, labels,
                                         min_queue_examples, batch_size,
                                         shuffle=False)




############ Generate synthetic data
import numpy as np
import pickle as pk
# for i in range(5):
#     a = np.random.rand(32, 32)
#     b = np.random.rand(100)
#     c = [a, b]
#     rfile = open('tmp/blocks_data/'+str(i), 'wb')
#     #tf.convert_to_tensor(c)
#     pk.dump(c, rfile)
#     rfile.close()
# sess = tf.Session()
# sess.run(inputs(False, 'tmp/blocks_data/', 10, sess=sess))

import h5py
hdf_file = h5py.File('tmp/blocks_data/dataset_1000_5.hdf5', 'r')
images = hdf_file.get('data')
labels = hdf_file.get('label')
print(labels.shape)
#############################################

# def read_blocks_bin(filename_queue):
#   """Reads and parses examples from CIFAR10 data files.
#
#   Recommendation: if you want N-way read parallelism, call this function
#   N times.  This will give you N independent Readers reading different
#   files & positions within those files, which will give better mixing of
#   examples.
#
#   Args:
#     filename_queue: A queue of strings with the filenames to read from.
#
#   Returns:
#     An object representing a single example, with the following fields:
#       height: number of rows in the result (32)
#       width: number of columns in the result (32)
#       depth: number of color channels in the result (3)
#       key: a scalar string Tensor describing the filename & record number
#         for this example.
#       label: an int32 Tensor with the label in the range 0..9.
#       uint8image: a [height, width, depth] uint8 Tensor with the image data
#   """
#
#   class BlocksRecord(object):
#     pass
#   result = BlocksRecord()
#
#   # Dimensions of the images in the Blocks dataset.
#   # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
#   # input format.
#   label_bytes = 1  # 2 for CIFAR-100
#   result.height = 32
#   result.width = 32
#   result.depth = 3
#   image_bytes = result.height * result.width * result.depth
#   # Every record consists of a label followed by the image, with a
#   # fixed number of bytes for each.
#   record_bytes = label_bytes + image_bytes
#
#   # Read a record, getting filenames from the filename_queue.  No
#   # header or footer in the CIFAR-10 format, so we leave header_bytes
#   # and footer_bytes at their default of 0.
#   reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
#   result.key, value = reader.read(filename_queue)
#
#   # Convert from a string to a vector of uint8 that is record_bytes long.
#   record_bytes = tf.decode_raw(value, tf.uint8)
#
#   # The first bytes represent the label, which we convert from uint8->int32.
#   result.label = tf.cast(
#       tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
#
#   # The remaining bytes after the label represent the image, which we reshape
#   # from [depth * height * width] to [depth, height, width].
#   depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
#                            [result.depth, result.height, result.width])
#   # Convert from [depth, height, width] to [height, width, depth].
#   result.uint8image = tf.transpose(depth_major, [1, 2, 0])
#
#   return result