import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def inputs(data_type, batch_size, num_epoch, num_threads):
    data_sets = input_data.read_data_sets('mnist_data')
    if data_type == 'train':
        input_images = tf.constant(data_sets.train.images)
        input_labels = tf.constant(data_sets.train.labels)
    else:
        input_images = tf.constant(data_sets.test.images)
        input_labels = tf.constant(data_sets.test.labels)

    image, label = tf.train.slice_input_producer(
        [input_images, input_labels], num_epochs=num_epoch, capacity = 10000)
    label = tf.cast(label, tf.int32)
    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=num_threads, capacity=10000)

    return images, labels

