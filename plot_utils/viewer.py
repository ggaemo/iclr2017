
import argparse
import itertools
import os

import numpy as np
import tensorflow as tf

# from tensorflow.python.client import timeline



parser = argparse.ArgumentParser()
parser.add_argument('-restore_model_dir', help='model_directory_to_restore', type=str)
parser.add_argument('-data_dir', help = 'data directory', type = str)
parser.add_argument('-test_data_size', help='test_data_size', type=int)
parser.add_argument('-test_batch_size', help='test_batch_size', type=int)

args = parser.parse_args()

model_dir = args.restore_model_dir
data_dir = args.data_dir
test_data_size = args.test_data_size
test_batch_size = args.test_batch_size


if not os.path.exists(data_dir+'/model/'+model_dir):
    raise FileExistsError()

if data_dir == 'cifar10':
    import cifar_10_inputs
    inputs = cifar_10_inputs.inputs
elif data_dir == 'higgs':
    import higgs_intputs
    inputs = higgs_intputs.inputs
elif data_dir =='mnist':
    from mnist import mnist_inputs

    inputs = mnist_inputs.inputs

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

with tf.Session(config=config) as sess:
    tf.set_random_seed(1)
    saver = tf.train.import_meta_graph(os.path.join(data_dir, 'summary',
                                                    model_dir+'.meta'))
    saver.restore(sess, os.path.join(data_dir, 'summary', model_dir))

    data = tf.get_collection('input_data')

    target = tf.get_collection('target_data')
    decoder_layer_output = tf.get_collection('decoder_layer_output')

    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    """
        2d-plot viewer
    """


    def get_part_change_value(a, num_total_config, dimension, i, j, value):
        tmp_i = i
        i = j
        j = tmp_i
        i = dimension * i
        j = dimension * j
        start_i = num_total_config * dimension - i
        start_j = j
        a[start_i - dimension:start_i, start_j:start_j + dimension] = value


    input_data, target_data, disentagled_layer_list, recon_data = \
        sess.run([
            test_model.input_data,
            test_model.target_data, test_model.disentagled_layer_list,
            test_model.decoder_layer_output])

    with tf.variable_scope('2d-plot'):

        span_interval = 15
        num_sample = 10

        sample_idx_list = np.random.choice(np.arange(input_data.shape[
                                                         0]),
                                           size=num_sample,
                                           replace=False)

        for sample_num, sample_idx in enumerate(sample_idx_list):
            original = input_data[sample_idx].reshape(28, 28)
            recon = recon_data[sample_idx].reshape(28, 28)
            target = target_data[sample_idx]

            np.save(os.path.join(data_dir, 'summary',
                                 model_dir, 'sample_recon_{}_step_{'
                                            '}.npy'.format(
                    sample_num, global_step_val)),
                    recon)
            np.save(os.path.join(data_dir, 'summary',
                                 model_dir,
                                 'sample_origin_{}_step_{'
                                 '}_target_{}.npy'.format(
                                     sample_num, global_step_val, target)),
                    original)

            for idx, disentagled_layer in enumerate(
                    disentagled_layer_list):
                # disentagled_layer이 axis의 기준이 되는 layer
                # other이 값이 fixed되어 있는 layer

                fixed_layer = disentagled_layer

                axis_layer = disentagled_layer_list[
                             :idx] + disentagled_layer_list[
                                     idx + 1:]
                axis_layer = axis_layer[0][sample_idx]

                min_val = axis_layer.min(0)
                max_val = axis_layer.max(0)
                i = np.linspace(min_val[0], max_val[0], span_interval)
                j = np.linspace(min_val[1], max_val[1], span_interval)
                ii, jj = np.meshgrid(i, j)
                range_input = np.stack((ii, jj), axis=2).reshape(-1, 2)
                # repeated_range_input = np.tile(range_input,
                #                                (disentagled_layer.shape[0], 1))

                # sample = disentagled_layer[sample_idx]

                canvas = np.zeros((28 * span_interval,
                                   28 * span_interval))
                fixed_repeated = np.tile(fixed_layer,
                                         (range_input.shape[0], 1))

                if idx == 0:
                    linspace_embedding = np.hstack([fixed_repeated,
                                                    range_input])
                else:
                    linspace_embedding = np.hstack([range_input,
                                                    fixed_repeated])

                fake_input = np.zeros(
                    test_model.encoder_layer_output.get_shape().as_list())

                fake_input[:span_interval * span_interval,
                :] = linspace_embedding

                z_space_output = sess.run(test_model.decoder_layer_output,
                                          feed_dict={
                                              test_model.encoder_layer_output: fake_input})
                z_space_output = z_space_output[
                                 :span_interval * span_interval]
                z_space_output = z_space_output.reshape(span_interval,
                                                        span_interval, 28,
                                                        28)

                for i, j in itertools.product(range(span_interval),
                                              range(span_interval)):
                    get_part_change_value(canvas, span_interval, 28,
                                          i, j, z_space_output[i, j,
                                                :])

                np.save(
                    os.path.join(data_dir, 'summary',
                                 model_dir,
                                 'sample_{}_layer_{}_step_{'
                                 '}.npy'.format(sample_num, idx,
                                                global_step_val)), canvas)

    except tf.errors.OutOfRangeError:
        logger.info('Done training -- epoch limit reached')
    except MaxPatienceError:
        logger.info('Done training -- max patience reached')
    finally:
        coord.request_stop()

    coord.join()
    sess.close()

