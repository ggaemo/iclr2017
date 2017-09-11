import argparse
import itertools
import os
import logging

import tensorflow as tf
import time

from network_regularization_2 import get_logger, RegPartModel
import regularization_losses

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
parser.add_argument('-huber_target', help='huber loss target', type=float, default = 1)
parser.add_argument('-huber_delta', help='huber loss delta', type=float, default = 1)
parser.add_argument('-num_target_class', help='number of classes in target', type=int,
                    default = 10)
parser.add_argument('-num_cycle', help='number of cycles for each epoch', type=int)
parser.add_argument('-option', help='optional specifications', type=str)

parser.add_argument('-n_gpu', help='number of gpus', type=int, default=3)

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
huber_target = args.huber_target
huber_delta = args.huber_delta
num_target_class = args.num_target_class
option = args.option
num_cycle = args.num_cycle
num_gpu = args.n_gpu

model_config = {'regType' : reg_type,
                'regModel' : '{0}_{1}'.format(num_partition_by_layer,
                                              reg_num_output_by_layer),
                'fc_bn' : is_bn,
                'reg_weight' : reg_weight,
                'delta' : huber_delta}

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

if reg_type == 'DotHuber':
    reg_loss = regularization_losses.DotHuber(reg_weight, huber_target, huber_delta)
elif reg_type == 'Dot':
    reg_loss = regularization_losses.Dot(reg_weight)
elif reg_type == 'Orthogonal':
    reg_loss = regularization_losses.Orthogonal(reg_weight)


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

opt = tf.train.AdamOptimizer()

grads_list = list()
with tf.name_scope('Train'):
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)


    train_inputs, train_targets = inputs('train', batch_size, num_epoch, 3)

    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [train_inputs, train_targets], capacity=2 * num_gpu)
    for i in range(num_gpu):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope('Model'):
                trn_model = RegPartModel(reg_loss, num_partition_by_layer,
                                         reg_num_output_by_layer, num_target_class,
                                         True, is_bn, train_inputs, train_targets)

                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(trn_model.loss)
                grads_list.append(grads)

    grads = average_gradients(grads_list)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)



with tf.name_scope('Test'):
    test_inputs, test_targets = inputs('test', batch_size, num_epoch, 3)
    with tf.variable_scope('Model', reuse=True):
        test_model = RegPartModel(reg_loss, num_partition_by_layer,
                                 reg_num_output_by_layer, num_target_class,
                                 False, is_bn, test_inputs, test_targets)


config=tf.ConfigProto(
        allow_soft_placement=True)
with tf.Session(config) as sess:

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
    max_patience = 5
    patience = 0
    try:
        prev = time.time()
        while not coord.should_stop():
            _, global_step = sess.run([trn_model.train_op, trn_model.global_step])

            if global_step % 500 == 0:
                summary_val = sess.run(merged)
                test_writer.add_summary(summary_val, global_step=global_step)

            if global_step % num_cycle == 0:

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
                                                                            num_cycle *
                                                                            batch_size /
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

