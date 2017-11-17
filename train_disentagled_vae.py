
import itertools
import re
import argparse
import os
import time
import pandas as pd
import numpy as np

import tensorflow as tf

from layerwise_discriminator import get_logger, Disentagled_VAE
# from tensorflow.python.client import timeline



parser = argparse.ArgumentParser()
parser.add_argument('-b', help = 'batch size', default=256, type=int)
parser.add_argument('-n', help = 'number of epochs', default=10000, type=int)
parser.add_argument('-enc', type=int, nargs='+')

parser.add_argument('-part_num', help = 'disentagled partition per layer', type=int)
parser.add_argument('-part_dim', help = 'disentagled dimension per layer', type=int)

parser.add_argument('-dec', type=int, nargs='+')

parser.add_argument('-bn', help = 'batch normalization in fully_connected',
                    action='store_true', default=False)
parser.add_argument('-data_dir', help = 'data directory', type = str)
parser.add_argument('-test_data_size', help='test_data_size', type=int)
parser.add_argument('-test_batch_size', help='test_batch_size', type=int)

parser.add_argument('-disen_act_fn', help='disentagled_activaion_fn', type=str)
parser.add_argument('-discrim', help='discrimiantor layer structure', type=int, nargs='+')


parser.add_argument('-num_cycle', help='number of cycles for each epoch', type=int)
parser.add_argument('-option', help='optional specifications', type=str)

parser.add_argument('-gpu_num', help='gpu number to operate', type=int, default=0)

parser.add_argument('-z_plot', help='whether to generate z-space interpolation plots or '
                                    'not', action='store_true')


class MaxPatienceError(Exception):
    pass

class OverfitError(Exception):
    pass

args = parser.parse_args()

batch_size = args.b
num_epoch = args.n
encoder_layer = args.enc
partition_num = args.part_num
partition_dim = args.part_dim
decoder_layer = args.dec
is_bn = args.bn
z_plot = args.z_plot
discrim_layer = args.discrim
disentagled_activation_fn = args.disen_act_fn
data_dir = args.data_dir
test_data_size = args.test_data_size
test_batch_size = args.test_batch_size

option = args.option
num_cycle = args.num_cycle
gpu_num = args.gpu_num

model_config = {'enc' : str(encoder_layer),
                'disent' : '{0}_{1}'.format(partition_num,
                                               partition_dim),
                'disent_fn' : disentagled_activation_fn,
                'dec': str(decoder_layer),
                'dis': str(discrim_layer),
                'is_bn' : is_bn}

for key, val in model_config.items():
    new_val = re.sub('[\[\]]', '', str(val))
    new_val = re.sub(', ', '-', new_val)
    model_config[key] = new_val


print(model_config)
if option:
    model_config['option'] = option

logger, model_dir = get_logger(data_dir, model_config)

if not os.path.exists(data_dir+'/model/'+model_dir):
    os.makedirs(data_dir+'/model/'+model_dir)

if data_dir == 'cifar-10':
    import cifar_10_inputs

    inputs = cifar_10_inputs.inputs
elif data_dir == 'higgs':
    import higgs_intputs
    inputs = higgs_intputs.inputs
elif data_dir =='mnist':
    import mnist_inputs
    inputs = mnist_inputs.inputs


if disentagled_activation_fn == 'relu':
    act_fn = tf.nn.relu
elif disentagled_activation_fn == 'tanh':
    act_fn = tf.nn.tanh
elif disentagled_activation_fn == 'sigmoid':
    act_fn = tf.sigmoid
elif disentagled_activation_fn == 'linear':
    act_fn = None


with tf.name_scope('Train'):
    train_inputs, train_targets = inputs('train', batch_size, num_epoch, 5)
    with tf.variable_scope('Model'):
        trn_model = Disentagled_VAE(
            encoder_layer, decoder_layer,
            partition_num, partition_dim,
            discrim_layer,
            True, is_bn,act_fn ,train_inputs, train_targets)

with tf.name_scope('Test'):
    test_inputs, test_targets = inputs('test', test_batch_size, num_epoch, 1)
    with tf.variable_scope('Model', reuse=True):
        test_model = Disentagled_VAE(encoder_layer, decoder_layer,
            partition_num, partition_dim,
            discrim_layer,
            False, is_bn, act_fn, test_inputs, test_targets)


    # with tf.name_scope('evaluate_z'):
    #     test_inputs, test_targets = inputs('test', 10, num_epoch, 5)
    #     with tf.variable_scope('Model', reuse=True):
    #         eval_model = Disentagled_VAE(encoder_layer, decoder_layer,
    #             partition_num, partition_dim,
    #             discrim_layer,
    #             False, is_bn, test_inputs)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True






with tf.Session(config=config) as sess:
    tf.set_random_seed(1)
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(logdir='{}/summary/{}/'.format(data_dir,
        model_dir), graph=sess.graph)
    saver = tf.train.Saver()
    tf.add_to_collection('input_data', test_model.input_data)
    tf.add_to_collection('target_data', test_model.target_data)
    tf.add_to_collection('decoder_layer_output', test_model.decoder_layer_output)
    tf.add_to_collection('train_op_discrim', test_model.train_op_discrim)
    tf.add_to_collection('train_op_gen', test_model.train_op_gen)
    tf.add_to_collection('encoder_layer_output', test_model.encoder_layer_output)

    # config = projector.ProjectorConfig()
    # embedding = config.embeddings.add()
    # embedding.tensor_name = trn_model.embedding_var.name
    # embedding.metadata_path = os.path.join(data_dir, 'metadata.tsv')


    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    best_loss = 1e4
    test_loss = [1e4, 1e4]
    max_patience = 50
    patience = 0

    # tf.get_default_graph().finalize()
    ''''
    finalize graph
    '''

    from tensorflow.contrib.tensorboard.plugins import projector
    config = projector.ProjectorConfig()


    try:
        prev = time.time()
        while not coord.should_stop():

            _, global_step_val = sess.run([trn_model.train_op_gen, trn_model.global_step])
            # _, global_step_val = sess.run([trn_model.train_op, trn_model.global_step])


            if global_step_val % 33 == 0:
                if test_loss[0] > 1e-4:
                    for _ in range(5):
                        sess.run(trn_model.train_op_discrim)

            # _, global_step_val = sess.run([trn_model.train_op, trn_model.global_step],
            #                               options=options,
            #                               run_metadata=run_metadata
            #                               )

            # if global_step_val % 100 == 0:
            #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #     with open('timeline_step_%d.json' % global_step_val, 'w') as f:
            #         f.write(chrome_trace)



            if global_step_val % 100 == 0:
                summary_val = sess.run(merged)
                test_writer.add_summary(summary_val, global_step=global_step_val)

            if global_step_val % 500 == 0:

                trn_loss_batch = sess.run([trn_model.discriminator_loss,
                                           trn_model.recon_loss])

                now = time.time()
                minutes = (now - prev) / 60
                message = 'took {} min, running at {} samples / min'.format(minutes,
                                                                            500
                                                                            * batch_size /
                                                                            minutes)

                test_loss = np.asarray([0.0, 0.0])

                for _ in range(test_data_size // test_batch_size):
                    test_loss_batch = sess.run([test_model.discriminator_loss,
                                                test_model.recon_loss])
                    test_loss += np.asarray(test_loss_batch)


                test_loss = test_loss * test_batch_size / test_data_size

                result = 'global_step: {} | trn_loss : {} test loss : {} '.format(
                    global_step_val, trn_loss_batch, test_loss)

                logger.info(result + '\n')
                logger.info(message + '\n')

                if global_step_val > 5000 and test_loss[1] > trn_loss_batch[1] * 1.5:
                    logger.info('Overfit')
                    raise OverfitError('Overfit')

                # l, acc = sess.run([test_model.loss, test_model.accuracy])
                # logger.info('reg epoch {} loss : {} accuracy {}'.format(global_step_val, l,
                #                                                         acc))

                if test_loss[1] < best_loss:
                    # z_value = np.array(sess.run(test_model.disentagled_layer_embedding))
                    # embed_tensor = tf.Variable(z_value, name='embed_disentagled_{}'.format(
                    #     global_step_val))
                    #
                    # num_disentagled_layer = embed_tensor.shape[0]
                    # sess.run(embed_tensor.initializer)
                    # embedding = config.embeddings.add()
                    # embedding.tensor_name = embed_tensor.name
                    # embedding.metadata_path = 'metadata_{}.tsv'.format(global_step_val)
                    # projector.visualize_embeddings(test_writer, config)
                    #
                    # # metadata = pd.DataFrame(np.), columns=['layer_num'])
                    # # z_columns = ['z_{}'.format(i) for i in range(z_value.shape[1])]
                    # # for col in z_columns:
                    # #     metadata[col] = None
                    # # metadata[z_columns] = z_value
                    #
                    # target_value = sess.run(test_model.target_data)
                    # target_tiled = np.tile(target_value, num_disentagled_layer)
                    # target_tiled[test_batch_size:] = target_tiled[test_batch_size:]+ 10
                    #
                    # metadata['target'] = target_tiled
                    #
                    #
                    # metadata.to_csv(os.path.join(data_dir, 'summary', model_dir,
                    #                              'metadata_{}.tsv'.format(global_step_val)), sep='\t',
                    #                 index=False)

                    embed_saver = tf.train.Saver()
                    embed_saver.save(sess, '{}/summary/{}/model.ckpt'.format(data_dir,
                                                                           model_dir),
                                     global_step=global_step_val)

                    """
                        2d-plot viewer
                    """

                    if z_plot:

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
                                    #disentagled_layer이 fixed의 기준이 되는 layer
                                    # other이 axis값이 되어 있는 layer

                                    fixed_layer = disentagled_layer[sample_idx]

                                    axis_layer = disentagled_layer_list[
                                            :idx] + disentagled_layer_list[
                                                    idx + 1:]
                                    axis_layer = axis_layer[0]

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
                                                            (range_input.shape[0],1))

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

                    print('better variable')

                    best_loss = test_loss[1]

                    # saver.save(sess, '{}/model/{}/model.ckpt'.format(data_dir,
                    #                         model_dir),
                    #            global_step=global_step_val)
                    patience = 0
                else:
                    patience += 1
                    if patience == max_patience:
                        logger.info('Max patience reached')
                        raise MaxPatienceError('Max patience')
                prev = time.time()

    except tf.errors.OutOfRangeError:
        logger.info('Done training -- epoch limit reached')
    except MaxPatienceError:
        logger.info('Done training -- max patience reached')
    finally:
        coord.request_stop()

    coord.join()
    sess.close()

