import pickle
import tensorflow as tf
import os
import pandas as pd
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten().tolist()))

def make_example(feature, label):
    feature = feature.tostring()

    ex = tf.train.Example(features=tf.train.Features(feature={
        'feature': _bytes_feature(feature),
        'label': _int64_feature(label.astype(int))
        }))

    return ex

def make_tfrecords():
    if os.path.exists('higgs_data/higgs_train.tfrecords'):
        print('higgs data.tfrecords exists')
    else:
        for data_type in ['train', 'test']:
            data = pd.read_hdf('higgs_data/HIGGS_{}.h5'.format(data_type), data_columns = False)
            filename = 'higgs_data/higgs_{}.tfrecords'.format(data_type)

            writer = tf.python_io.TFRecordWriter(filename)

            for row in data.values:
                feature = row[1:22]
                label = row[0]
                ex = make_example(feature, label)
                writer.write(ex.SerializeToString())
            writer.close()
            print('tfrecord made')


def read_and_decode(filename_queue):
    print('Reading and Decoding')
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features =  {'feature' : tf.FixedLenFeature([], tf.string),
                 'label' : tf.FixedLenFeature([], tf.int64)}

    data_parsed = tf.parse_single_example(
        serialized=serialized_example,
        features=features
        )

    label = tf.cast(data_parsed['label'], tf.int32)

    feature = tf.cast(tf.decode_raw(data_parsed['feature'], tf.float64), tf.float32)

    feature.set_shape([21])

    data_dict = {'feature' : feature, 'label' : label}

    return data_dict


def inputs(data_type, batch_size, num_epochs, num_threads=1):

    filename = ['higgs_data/higgs_{}.tfrecords'.format(data_type)]
    filename_queue = tf.train.string_input_producer(filename, num_epochs)
    reader_output = read_and_decode(filename_queue)

    batch = tf.train.batch([reader_output['feature'],
                            reader_output['label']
                            ],
                             batch_size, allow_smaller_final_batch=False,
                           capacity = batch_size * 2, num_threads=num_threads)
    return batch


def test():
    make_tfrecords()
    input_data_stream = inputs('test', 128, 10)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        while not coord.should_stop():
            a = sess.run(input_data_stream)
            print(a)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join()
    sess.close()
if __name__ == '__main__':
    test()

make_tfrecords()