
import tensorflow as tf
import os
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def make_example(image, latent):
    image = image.tostring()
    latent = latent.astype(np.float32)
    ex = tf.train.Example(features=tf.train.Features(feature={
        'img': _bytes_feature(image),
        'latent' : _float_feature([latent])
        }))

    return ex

def make_tfrecords(random=False):
    for i in [0, 1,2]:
        if random:
            filename = 'dsprite/random_data/dsprite_{}.tfrecords'.format(i)
        else:
            filename = 'dsprite/data/dsprite_{}.tfrecords'.format(i)

        if os.path.exists(filename):
            print(filename, 'exists')
            continue
        else:
            if random:
                data = np.load('dsprite/random_data/data_batch_{}.npz'.format(i))
            else:
                data = np.load('dsprite/data/data_batch_{}.npz'.format(i))

            writer = tf.python_io.TFRecordWriter(filename)
            for image, latent in zip(data['img'], data['latent']):
                ex = make_example(image, latent)
                writer.write(ex.SerializeToString())
            writer.close()
            print('tfrecord made')

def make_tfrecords_single():

    filename = 'dsprite/random_data/dsprite_aligned.tfrecords'

    if os.path.exists(filename):
        print(filename, 'exists')
    else:
        data = np.load('dsprite/random_data/data_aligned.npz')

        writer = tf.python_io.TFRecordWriter(filename)
        for image, latent in zip(data['img'], data['latent']):
            latent = latent[0]
            ex = make_example(image, latent)
            writer.write(ex.SerializeToString())
        writer.close()
        print('tfrecord made')

def read_and_decode(example_proto):
    print('Reading and Decoding')
    features =  {'img' : tf.FixedLenFeature([], tf.string),
                 'latent' : tf.FixedLenFeature([1], tf.float32)}


    data_parsed = tf.parse_single_example(
        example_proto,
        features=features
        )


    image = tf.decode_raw(data_parsed['img'], tf.uint8)
    image = tf.reshape(image, [64,  64, 1])

    image = tf.cast(image, tf.float32)

    latent = data_parsed['latent']

    data_dict = {'img' : image,
                 'latent' : latent}
    return data_dict


def inputs(batch_size, num_parallel_calls):

    # filenames = ['dsprite/random_data/dsprite_{}.tfrecords'.format(i) for i in
    #                  range(3)]
    filenames = ['dsprite/data/dsprite_{}.tfrecords'.format(i) for i in
                 range(3)]


    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_and_decode, num_parallel_calls)
    dataset = dataset.shuffle(buffer_size =20000)
    dataset = dataset.batch(batch_size)
    dataset.repeat()
    iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                       dataset.output_shapes)
    next_element = iterator.get_next()
    init_op = iterator.make_initializer(dataset)

    return next_element, init_op

def inputs_single_file(batch_size, num_parallel_calls):
    filenames = 'dsprite/random_data/dsprite_aligned.tfrecords'

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_and_decode, num_parallel_calls)
    dataset = dataset.batch(batch_size)
    dataset.repeat()
    iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types,
                                       dataset.output_shapes)
    next_element = iterator.get_next()
    init_op = iterator.make_initializer(dataset)

    return next_element, init_op


def test():
    make_tfrecords()
    input_data_stream, init_op = inputs('train', 128, 10, 3)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(init_op)
    try:
        while True:
            a = sess.run(input_data_stream)
            print(a['image'].shape)
            import matplotlib.pyplot as plt
            plt.imshow(a['image'][0].reshape(64, 64))
            plt.show()
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    sess.close()

if __name__ == '__main__':
    test()

# make_tfrecords(random=False)
# make_tfrecords(random=True)
make_tfrecords_single()