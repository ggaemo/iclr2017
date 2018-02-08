import pickle
import tensorflow as tf
import os, json, re, itertools, collections
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))

def make_example(image, label):

    image = image.tostring()
    ex = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image),
        'label': _int64_feature(label)}))
    return ex

def make_tfrecords():
    for i in [1,2,3,4,5, 'test']:
        filename = 'cifar-10_data/cifar-10_{}.tfrecords'.format(i)
        if os.path.exists(filename):
            print(filename, 'exists')
            continue
        else:
            with open('cifar-10_data/data_batch_{}'.format(i), 'rb') as f:
                pickled_data = pickle.load(f, encoding='bytes')

            writer = tf.python_io.TFRecordWriter(filename)

            for image, label in zip(pickled_data[b'data'], pickled_data[\
                    b'labels']):
                ex = make_example(image, label)
                writer.write(ex.SerializeToString())
            writer.close()
            print('tfrecord made')


def read_and_decode(filename_queue):
    print('Reading and Decoding')
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features =  {'image' : tf.FixedLenFeature([], tf.string),
                 'label' : tf.FixedLenFeature([], tf.int64)}

    data_parsed = tf.parse_single_example(
        serialized=serialized_example,
        features=features
        )

    image = tf.decode_raw(data_parsed['image'], tf.uint8)
    image.set_shape([32 * 32 * 3])

    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    label = tf.cast(data_parsed['label'], tf.int32)

    data_dict = {'image' : image, 'label' : label}
    return data_dict


def inputs(data_type, batch_size, num_epochs, num_threads=1):

    if data_type == 'train':
        filename = list()
        for i in [1,2,3,4,5]:
            filename.append('cifar-10_data/cifar-10_{}.tfrecords'.format(i))
    elif data_type == 'test':
        filename = ['cifar-10_data/cifar-10_test.tfrecords']
    filename_queue = tf.train.string_input_producer(filename, num_epochs)
    reader_output = read_and_decode(filename_queue)

    batch = tf.train.batch([reader_output['image'],
                            reader_output['label']
                            ],
                             batch_size, allow_smaller_final_batch=False,
                           capacity = batch_size * 5, num_threads=num_threads)
    return batch


def test():
    make_tfrecords()
    input_data_stream = inputs('train', 128, 10)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        while not coord.should_stop():
            a, b = sess.run(input_data_stream)
            print(a)
            print(b)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join()
    sess.close()

if __name__ == '__main__':
    test()
make_tfrecords()