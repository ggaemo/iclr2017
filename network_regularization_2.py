import itertools
import os
import logging

import tensorflow as tf

def get_logger(data_dir, model_config):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # create a file handler
    model_dir = '_'.join(str(key) + '-' + str(val) for key, val in model_config.items())

    if not os.path.exists(data_dir+'/log/' + model_dir):
        os.makedirs(data_dir+'/log/' + model_dir)

    if not os.path.exists(data_dir + '/summary/' + model_dir):
        os.makedirs(data_dir+'/summary/' + model_dir)

    handler = logging.FileHandler(data_dir+'/log/' + model_dir +'/log.txt')
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger, model_dir


class RegPartModel():
    def __init__(self, reg_loss, num_partition_by_layer, num_outputs_by_layer,
                 num_target_class,
                 is_training, is_bn, input_data, target):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        def fully_connected_bn(layer_input, num_outputs):
            layer = tf.contrib.layers.fully_connected(layer_input, num_outputs)
            layer = tf.contrib.layers.batch_norm(layer, is_training = is_training,
                                                 fused=True)
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

        def layer_attention(layer_list):
            layer_output = list()
            for layer in layer_list:
                # layer_dim = layer.shape.as_list()[1]
                attention = tf.contrib.layers.fully_connected(layer, 1,
                                                            activation_fn=tf.sigmoid)
                # attention = tf.contrib.layers.fully_connected(tmp, 1,
                #                                               activation=tf.sigmoid)
                # layer_output += layer * attention
                layer_output.append(layer * attention)
            layer_output = tf.concat(layer_output, axis=1)
            return layer_output




        layer_list = list()
        layer_list.append(input_data)
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
                # activation = tf.reduce_mean(tf.reduce_mean(tf.cast(tf.greater(layer, 0.0),
                #                                          tf.float32),axis=1))

                activation = 1.0 - tf.nn.zero_fraction(layer)
                tf.summary.scalar('layer_{}_partition_{}_activation_ratio'.format(
                    layer_num, idx), activation)


            reg_loss_list.append(reg_loss.loss(layer))

            # layer_concat = tf.concat(layer, axis=1)
            layer_concat = layer_attention(layer)
            layer_list.append(layer_concat)

        output_layer = tf.contrib.layers.fully_connected(layer_list[-1],
                                                         num_outputs=num_target_class)

        self.reg_loss = tf.reduce_mean(reg_loss_list)
        self.xent_loss = tf.losses.sparse_softmax_cross_entropy(target, output_layer)
        self.loss =  self.xent_loss + self.reg_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss,
                                                              global_step=self.global_step)

        with tf.variable_scope('metrics'):
            correct_pred = tf.equal(target, tf.cast(tf.argmax(output_layer, 1), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.histogram('output_layer', output_layer)
        tf.summary.scalar('xent_loss', self.xent_loss)
        tf.summary.scalar('reg_loss', self.reg_loss)
        tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)


