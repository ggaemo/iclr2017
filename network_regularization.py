import argparse
import itertools
import os
import logging

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import gen_math_ops


parser = argparse.ArgumentParser()
parser.add_argument('-b', help = 'batch size', default=256, type=int)
parser.add_argument('-n', help = 'number of epochs', default=10000, type=int)
parser.add_argument('-reg', help = 'regularization type', type=str)
parser.add_argument('-np', help = 'reg NN output partition per layer', type=int, nargs='+')
parser.add_argument('-no', help = 'reg NN output dimension per layer', type=int, nargs='+')
parser.add_argument('-bn', help = 'batch normalization in fully_connected',
                    action='store_true')

def get_logger(model_config):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # create a file handler
    model_dir = '_'.join(str(key) + '-' + str(val) for key, val in model_config.items())

    if not os.path.exists('log/' + model_dir):
        os.makedirs('log/' + model_dir)

    if not os.path.exists('summary/' + model_dir):
        os.makedirs('summary/' + model_dir)

    handler = logging.FileHandler('log/' + model_dir +'/log.txt')
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger, model_dir

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

inputs = tf.placeholder(tf.float32, [None, 784], name = 'input')
target = tf.placeholder(tf.float32, [None, 10], name = 'target')
is_training = tf.placeholder(tf.bool, name='bn_phase')

class SimpleModel():
    def __init__(self, inputs, num_outputs):

        layer_1 = tf.contrib.layers.fully_connected(inputs, num_outputs)

        output_layer = tf.contrib.layers.fully_connected(layer_1, num_outputs = 10)

        self.loss = tf.losses.softmax_cross_entropy(target, output_layer)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        with tf.variable_scope('metrics'):
            correct_pred = tf.equal(tf.argmax(target, 1), tf.argmax(output_layer, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.histogram('layer_1', layer_1)

        # tf.summary.histogram('activation', tf.cast(tf.equal(layer_1, 0.0),tf.float32))
        zeros = tf.zeros_like(layer_1, dtype=tf.float32)
        tf.summary.histogram('activation',
                             tf.cast(gen_math_ops.approximate_equal(layer_1, zeros, 1e-4),
                                     tf.float32))

        tf.summary.histogram('output_layer', output_layer)
        tf.summary.scalar('xent_loss', self.loss)
        # tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

class RegPartModel():
    def __init__(self, inputs, num_partition_by_layer, num_outputs_by_layer,
                 is_training, is_bn, target, reg_type):

        def fully_connected_bn(layer_input, num_outputs):
            layer = tf.contrib.layers.fully_connected(layer_input, num_outputs)
            layer = tf.contrib.layers.batch_norm(layer, is_training = is_training)
            '''
            이런 방식으로 하는게 맞나..??? (is_training을 init에서 나온 scope에서 가져오독 하는것)
            '''
            return layer

        def fully_connected(layer_input, num_outputs):
            layer = tf.contrib.layers.fully_connected(layer_input, num_outputs)
            return layer

        def make_partitioned_layer(layer_input, num_partition, num_outputs, is_bn):
            partitioned_layer_list = list()
            for _ in range(num_partition):
                if is_bn:
                    partitioned_layer_list.append(
                        fully_connected_bn(layer_input, num_outputs))
                else:
                    partitioned_layer_list.append(
                        fully_connected(layer_input, num_outputs))

            return partitioned_layer_list

        def reg_loss_dot(layer_list):
            reg_loss = 0
            for i, j in itertools.combinations(layer_list, 2):
                reg_loss += tf.reduce_mean(tf.multiply(i, j))
            return reg_loss

        def reg_loss_dot_huber(layer_list):
            reg_loss = 0
            # neg_ones = -tf.ones(layer_list.shape[0])
            neg_ones = -tf.gather(tf.zeros_like(layer_list[0]), 0, axis=1)
            for i, j in itertools.combinations(layer_list, 2):
                dot_prod = tf.reduce_mean(tf.multiply(i, j), axis=1)
                reg_loss += tf.losses.huber_loss(neg_ones, dot_prod)
            return reg_loss

        def reg_loss_orthogonal(layer_list):
            reg_loss = 0
            for i, j in itertools.combinations(layer_list, 2):
                reg_loss += tf.sqrt(tf.reduce_mean(tf.square(tf.multiply(i, j))))
            return reg_loss

        layer_list = list()
        layer_list.append(inputs)
        layer_num_range = range(len(num_partition_by_layer))
        reg_loss_list = list()
        for layer_num, num_partition, num_outputs in zip(layer_num_range,
                                                         num_partition_by_layer,
                                                         num_outputs_by_layer):
            layer_inputs = layer_list[-1]
            layer = make_partitioned_layer(layer_inputs, num_partition, num_outputs, is_bn)

            for idx, partitioned_layer in enumerate(layer):
                tf.summary.histogram('layer_{}_partition_{}'.format(layer_num, idx),
                                     partitioned_layer)
                activation = tf.cast(tf.greater(layer, 0.0), tf.int32)
                tf.summary.histogram('layer_{}_partition_{}_activation'.format(
                    layer_num, idx), activation)

            if reg_type == 'dotProduct':
                reg_loss_list.append(reg_loss_dot(layer))
            elif reg_type == 'dotOrthogonalProduct':
                reg_loss_list.append(reg_loss_orthogonal(layer))
            elif reg_type == 'dotProductHuber':
                reg_loss_list.append(reg_loss_dot_huber(layer))
            else:
                raise AttributeError('unidentified regularzation method')

            layer_concat = tf.concat(layer, axis=1)
            layer_list.append(layer_concat)

        output_layer = tf.contrib.layers.fully_connected(layer_list[-1],num_outputs=10)

        self.reg_loss = tf.reduce_mean(reg_loss_list)
        self.xent_loss = tf.losses.softmax_cross_entropy(target, output_layer)
        self.loss =  self.xent_loss + self.reg_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        with tf.variable_scope('metrics'):
            correct_pred = tf.equal(tf.argmax(target, 1), tf.argmax(output_layer, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.histogram('output_layer', output_layer)
        tf.summary.scalar('xent_loss', self.xent_loss)
        tf.summary.scalar('reg_loss', self.reg_loss)
        tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

args = parser.parse_args()

batch_size = args.b
num_epoch = args.n
num_partition_by_layer = args.np
reg_num_output_by_layer = args.no
is_bn = args.bn
reg_type = args.reg

# model_config = {'regType' : 'simpleModel',
#                 'simpleModel' : '{}'.format(simple_num_output)}

model_config = {'regType' : reg_type,
                'regModel' : '{0}_{1}'.format(num_partition_by_layer,
                                              reg_num_output_by_layer),
                'fc_bn' : is_bn}

logger, model_dir = get_logger(model_config)

# with tf.variable_scope('simple_model'):
#     simple_model = SimpleModel(inputs, simple_num_output)

with tf.variable_scope('reg_model'):
    reg_model = RegPartModel(inputs, num_partition_by_layer, reg_num_output_by_layer,
                             is_training, is_bn, target, reg_type)

with tf.Session() as sess:
    tf.set_random_seed(1)
    merged = tf.summary.merge_all()
    # trn_writer = tf.summary.FileWriter(logdir='summary/{}/train'.format(model_dir),
    #                                    graph=sess.graph)
    test_writer = tf.summary.FileWriter(logdir='summary/{}/test'.format(model_dir),
                                        graph=sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(num_epoch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(reg_model.train_op
                 , feed_dict={inputs :batch_x, target: batch_y, is_training : True})

        if i % 100 == 0:
            summary_val = sess.run(merged, feed_dict={inputs : mnist.test.images,
                                                      target: mnist.test.labels ,
                                                      is_training : False})
            test_writer.add_summary(summary_val, global_step=i)

        if i % 1000 == 0:
            # l, acc = sess.run([simple_model.loss, simple_model.accuracy],
            #                   feed_dict={inputs : mnist.test.images,
            #                              target: mnist.test.labels})
            # logger.info('sm epoch {} loss : {} accuracy {}'.format(i, l, acc))

            l, acc = sess.run([reg_model.loss, reg_model.accuracy],
                              feed_dict={inputs: mnist.test.images,
                                         target: mnist.test.labels,
                                         is_training : False
                                         })
            logger.info('reg epoch {} loss : {} accuracy {}'.format(i, l, acc))