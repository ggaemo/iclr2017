import argparse
import itertools
import os
import logging

import tensorflow as tf
import time

from network_regularization_2 import get_logger, RegPartModel


parser = argparse.ArgumentParser()
parser.add_argument('-b', help = 'batch size', default=256, type=int)
parser.add_argument('-n', help = 'number of epochs', default=10000, type=int)
parser.add_argument('-reg', help = 'regularization type', type=str)
parser.add_argument('-np', help = 'reg NN output partition per layer', type=int, nargs='+')
parser.add_argument('-no', help = 'reg NN output dimension per layer', type=int, nargs='+')
parser.add_argument('-reg_w', help = 'regularzation loss weight(lambda)', type=float)
parser.add_argument('-bn', help = 'batch normalization in fully_connected', action='store_true')
parser.add_argument('-data_dir', help = 'data directory', type = str)
parser.add_argument('-test_data_size', help='test_data_size', type=int)
parser.add_argument('-test_batch_size', help='test_batch_size', type=int)
parser.add_argument('-huber_delta', help='huber loss delta', type=float, default = 1)
parser.add_argument('-num_target_class', help='number of classes in target', type=int,
                    default = 10)

class MaxPatienceError(Exception):
    pass

args = parser.parse_args()

batch_size = args.b
num_epoch = args.n
num_partition_by_layer = args.np
reg_num_output_by_layer = args.no
is_bn = args.bn
reg_type = args.reg
reg_weight = args.reg_w
data_dir = args.data_dir
test_data_size = args.test_data_size
test_batch_size = args.test_batch_size
huber_delta = args.huber_delta
num_target_class = args.num_target_class


model_config = {'regType' : reg_type,
                'regModel' : '{0}_{1}'.format(num_partition_by_layer,
                                              reg_num_output_by_layer),
                'fc_bn' : is_bn,
                'reg_weight' : reg_weight,
                'delta' : huber_delta}

logger, model_dir = get_logger(data_dir, model_config)

if not os.path.exists(data_dir+'/model/'+model_dir):
    os.makedirs(data_dir+'/model/'+model_dir)

if data_dir == 'cifar-10':
    import cifar_10_inputs

    inputs = cifar_10_inputs.inputs
elif data_dir == 'higgs':
    import higgs_intputs
    inputs = higgs_intputs.inputs

with tf.name_scope('Train'):
    train_inputs, train_targets = inputs('train', batch_size, num_epoch, 3)
    with tf.variable_scope('Model', reuse=None):
        trn_model = RegPartModel(train_inputs, num_partition_by_layer,
                                 reg_num_output_by_layer,
                                 True, is_bn, train_targets, reg_type, reg_weight,
                                 num_target_class, huber_delta)

with tf.name_scope('Test'):
    test_inputs, test_targets = inputs('test', batch_size, num_epoch, 3)
    with tf.variable_scope('Model', reuse=True):
        test_model = RegPartModel(test_inputs, num_partition_by_layer,
                                  reg_num_output_by_layer,
                                  False, is_bn, test_targets, reg_type, reg_weight,
                                  num_target_class, huber_delta)

    #
    # input_data_stream = inputs(128, 10)
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess, coord)
    # try:
    #     while not coord.should_stop():
    #         a, b = sess.run(input_data_stream)
    #         print(a)
    #         print(b)
    # except tf.errors.OutOfRangeError:
    #     print('Done training -- epoch limit reached')
    # finally:
    #     coord.request_stop()
    #
    # coord.join()
    # sess.close()


with tf.Session() as sess:

    tf.set_random_seed(1)
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(logdir='{}/summary/{}/test'.format(data_dir,
        model_dir),
                                        graph=sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)


    best_loss = 1e4
    max_patience = 3
    patience = 0
    try:
        prev = time.time()
        while not coord.should_stop():
            _, global_step = sess.run([trn_model.train_op, trn_model.global_step])

            if global_step % 500 == 0:
                summary_val = sess.run(merged)
                test_writer.add_summary(summary_val, global_step=global_step)

            if global_step % 1000 == 0:

                trn_loss, trn_acc = sess.run([trn_model.loss,
                                              trn_model.accuracy])
                now = time.time()
                test_loss = 0
                test_acc = 0
                for _ in range(test_data_size // test_batch_size):
                    test_loss_batch, test_acc_batch = sess.run([test_model.loss,
                                                                test_model.accuracy])
                    test_loss += test_loss_batch
                    test_acc += test_acc_batch

                test_loss = test_loss * test_batch_size / test_data_size
                test_acc = test_acc * test_batch_size / test_data_size

                minutes = (now - prev) / 60
                result = 'global_step: {} | trn_loss : {} trn_acc : {} test loss : {} test acc : {}'.format(
                    global_step, trn_loss,
                    trn_acc, test_loss, test_acc)
                message = 'took {} min, running at {} samples / min'.format(minutes,
                                                                            test_data_size /
                                                                            minutes)
                logger.info(result + '\n')
                logger.info(message + '\n')

                prev = time.time()

                # l, acc = sess.run([test_model.loss, test_model.accuracy])
                # logger.info('reg epoch {} loss : {} accuracy {}'.format(global_step, l, acc))

                if test_loss < best_loss:
                    best_loss = test_loss
                    saver.save(sess, data_dir + '/model/' + model_dir + '/model.ckpt',
                               global_step=global_step)
                    patience = 0
                else:
                    patience += 1
                    if patience == max_patience:
                        logger.info('Max patience reached')
                        raise MaxPatienceError('Max patience')

    except tf.errors.OutOfRangeError:
        logger.info('Done training -- epoch limit reached')
    except MaxPatienceError:
        logger.info('Done training -- max patience reached')
    finally:
        coord.request_stop()

    coord.join()
    sess.close()

