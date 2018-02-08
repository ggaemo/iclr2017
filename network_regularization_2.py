import itertools
import os
import logging

import tensorflow as tf

def get_logger(data_dir, model_config):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    keys_to_remove = [x for x in model_config.keys() if not x]
    for key in keys_to_remove:
        model_config.pop(key)

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
    def __init__(self, reg_loss, normal_layer, num_partition_by_layer,
                 num_outputs_by_layer,
                 num_target_class,
                 is_training, is_bn, input_data, target):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        def fully_connected_bn(layer_input, num_outputs):
            bn_layer = tf.contrib.layers.batch_norm(layer_input, is_training=is_training,
                                                    fused=True, center=True, scale=True)
            layer = tf.contrib.layers.fully_connected(bn_layer, num_outputs)

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
                    partitioned_layer_list.append(fully_connected_bn(layer_input, num_outputs))
                else:
                    partitioned_layer_list.append(
                        fully_connected(layer_input, num_outputs))

            return partitioned_layer_list

        def layer_attention(layer_list):
            layer_output = list()
            for layer in layer_list:
                layer_dim = layer.shape.as_list()[1]
                bottle_neck_dim = int(layer_dim / 8)
                hidden = tf.contrib.layers.fully_connected(layer, bottle_neck_dim)
                attention = tf.contrib.layers.fully_connected(hidden, 1,
                                                            activation_fn=tf.sigmoid)
                # attention = tf.contrib.layers.fully_connected(tmp, 1,
                #                                               activation=tf.sigmoid)
                # layer_output += layer * attention
                layer_output.append(layer * attention)
            layer_output = tf.concat(layer_output, axis=1)
            return layer_output

        with tf.variable_scope('normal_layers'):
            normal_layer_list = list()
            normal_layer_list.append(input_data)
            for layer_num, layer_dim in enumerate(normal_layer):
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = normal_layer_list[-1]
                    normal_layer_list.append(fully_connected_bn(layer_inputs, layer_dim))

        if reg_loss:
            with tf.variable_scope('partitioned_layers'):
                layer_list = list()
                layer_list.append(normal_layer_list[-1])

                layer_num_range = range(len(num_partition_by_layer))
                reg_loss_list = list()
                for layer_num, num_partition, num_outputs in zip(layer_num_range,
                                                                 num_partition_by_layer,
                                                                 num_outputs_by_layer):
                    with tf.variable_scope('layer_{}'.format(layer_num)):
                        layer_inputs = layer_list[-1]
                        layer = make_partitioned_layer(layer_inputs,
                                                               num_partition, num_outputs, is_bn)
                        reg_loss_list.append(reg_loss.loss(layer))
                        layer_concat = tf.concat(layer, axis=1)
                        layer_list.append(layer_concat)

        with tf.variable_scope('layer_activations'):
            for layer_num, layer in enumerate(normal_layer_list):
                if layer_num == 0:
                    continue # input이 normal_layer_list의 첫번째로 들어가 있으므로 뺀다.
                activation = 1.0 - tf.nn.zero_fraction(layer)
                tf.summary.histogram('norm_{}_act_val'.format(layer_num), layer)
                tf.summary.scalar('norm_{}_act_ratio'.format(layer_num),
                                  activation)

            if reg_loss:
                for idx, partitioned_layer in enumerate(layer_list):
                    tf.summary.histogram('reg_{}_part_{}_act_val'.format(layer_num, idx),
                                         partitioned_layer)
                    activation = 1.0 - tf.nn.zero_fraction(partitioned_layer)
                    tf.summary.scalar('reg_{}_part_{}_act_ratio'.format(
                        layer_num, idx), activation)

        # layer_concat = layer_attention(layer)
        if reg_loss:
            before_logit_layer = layer_list[-1]
        else:
            before_logit_layer = normal_layer_list[-1]

        output_layer = tf.contrib.layers.fully_connected(before_logit_layer,
                                                         num_outputs=num_target_class)

        with tf.variable_scope('loss'):
            if reg_loss:
                with tf.variable_scope('regularization_loss'):
                    self.reg_loss = tf.reduce_mean(reg_loss_list)
            else:
                self.reg_loss = 0.0

            with tf.variable_scope('cross_entropy_loss'):
                self.xent_loss = tf.losses.sparse_softmax_cross_entropy(target, output_layer)
            self.loss =  self.xent_loss + self.reg_loss

        with tf.variable_scope('metrics'):

            if num_target_class == 2:
                softmax = tf.nn.softmax(output_layer)
                self.prob_1 = tf.gather(softmax, 1, axis=1)
                self.auc, _auc_update_op = tf.metrics.auc(target, self.prob_1,
                                                          updates_collections='auc_update')

            prediction = tf.cast(tf.argmax(output_layer, 1), tf.int32)
            correct_pred = tf.equal(target, prediction)

            if num_target_class == 2:
                with tf.control_dependencies([_auc_update_op]):
                    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            else:
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss,
                                                              global_step=self.global_step)

        tf.summary.histogram('output_layer', output_layer)
        tf.summary.scalar('xent_loss', self.xent_loss)
        if reg_loss:
            tf.summary.scalar('reg_loss', self.reg_loss)
        # tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        if num_target_class == 2:
            with tf.device('cpu:0'):
                tf.summary.scalar('auc', self.auc)


class RegPartWeightModel():
    def __init__(self, reg_loss, normal_layer, reg_layer,
                 num_target_class,
                 is_training, is_bn, input_data, target):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        def fully_connected_bn(layer_input, num_outputs):
            bn_layer = tf.contrib.layers.batch_norm(layer_input, is_training=is_training,
                                                    fused=True, center=True, scale=True)
            layer = tf.contrib.layers.fully_connected(bn_layer, num_outputs,
                                                      variables_collections=['W_b'])

            '''
            이런 방식으로 하는게 맞나..??? (is_training을 init에서 나온 scope에서 가져오독 하는것)
            '''
            return layer

        def fully_connected(layer_input, num_outputs):
            layer = tf.contrib.layers.fully_connected(layer_input, num_outputs,
                                                      variables_collections = ['W_b'])
            return layer

        def make_partitioned_layer(layer_input, num_partition, num_outputs, is_bn):
            partitioned_layer_list = list()

            for _ in range(num_partition):
                if is_bn:
                    partitioned_layer_list.append(fully_connected_bn(layer_input, num_outputs))
                else:
                    partitioned_layer_list.append(
                        fully_connected(layer_input, num_outputs))

            return partitioned_layer_list

        def layer_attention(layer_list):
            layer_output = list()
            for layer in layer_list:
                layer_dim = layer.shape.as_list()[1]
                bottle_neck_dim = int(layer_dim / 8)
                hidden = tf.contrib.layers.fully_connected(layer, bottle_neck_dim, )
                attention = tf.contrib.layers.fully_connected(hidden, 1,
                                                            activation_fn=tf.sigmoid)
                # attention = tf.contrib.layers.fully_connected(tmp, 1,
                #                                               activation=tf.sigmoid)
                # layer_output += layer * attention
                layer_output.append(layer * attention)
            layer_output = tf.concat(layer_output, axis=1)
            return layer_output

        with tf.variable_scope('normal_layers'):
            normal_layer_list = list()
            normal_layer_list.append(input_data)
            for layer_num, layer_dim in enumerate(normal_layer):
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = normal_layer_list[-1]
                    normal_layer_list.append(fully_connected_bn(layer_inputs, layer_dim))
        if reg_loss:
            with tf.variable_scope('regularized_layers'):
                reg_layer_list = list()
                reg_layer_list.append(normal_layer_list[-1])
            reg_loss_list = list()
            for layer_num, layer_dim in enumerate(reg_layer):
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = reg_layer_list[-1]
                    reg_layer_list.append(fully_connected_bn(layer_inputs, layer_num))
                W = tf.get_collection('W_b','Model/partitioned_layers/layer_{}/'.format(layer_num))
                W = [x for x in W if 'weight' in x.name][0]
                reg_loss_list.append(reg_loss.loss(W))


        # if reg_loss:
        #     with tf.variable_scope('regularized_layers'):
        #         layer_list = list()
        #         layer_list.append(normal_layer_list[-1])
        #
        #         layer_num_range = range(len(num_partition_by_layer))
        #         reg_loss_list = list()
        #         for layer_num, num_partition, num_outputs in zip(layer_num_range,
        #                                                          num_partition_by_layer,
        #                                                          num_outputs_by_layer):
        #             with tf.variable_scope('layer_{}'.format(layer_num)):
        #                 layer_inputs = layer_list[-1]
        #                 layer = make_partitioned_layer(layer_inputs,
        #                                                        num_partition, num_outputs, is_bn)
        #                 W_list = tf.get_collection('W_b',
        #                                            'Model/partitioned_layers/layer_{}/'.format(layer_num))
        #                 W_list = [x for x in W_list if 'weight' in x.name]
        #                 reg_loss_list.append(reg_loss.loss(W_list))
        #                 layer_concat = tf.concat(layer, axis=1)
        #                 layer_list.append(layer_concat)

        with tf.variable_scope('layer_activations'):
            for layer_num, layer in enumerate(normal_layer_list):
                if layer_num == 0:
                    continue # input이 normal_layer_list의 첫번째로 들어가 있으므로 뺀다.
                activation = 1.0 - tf.nn.zero_fraction(layer)
                tf.summary.histogram('norm_{}_act_val'.format(layer_num), layer)
                tf.summary.scalar('norm_{}_act_ratio'.format(layer_num),
                                  activation)

            if reg_loss:
                for layer_num, reg_layer in enumerate(reg_layer_list):
                    tf.summary.histogram('reg_{}_act_val'.format(layer_num, layer_num),
                                         reg_layer)
                    activation = 1.0 - tf.nn.zero_fraction(reg_layer)
                    tf.summary.scalar('reg_{}_act_ratio'.format(
                        layer_num), activation)

        # layer_concat = layer_attention(layer)
        if reg_loss:
            before_logit_layer = reg_layer_list[-1]
        else:
            before_logit_layer = normal_layer_list[-1]

        output_layer = tf.contrib.layers.fully_connected(before_logit_layer,
                                                         num_outputs=num_target_class)

        with tf.variable_scope('loss'):
            if reg_loss:
                with tf.variable_scope('regularization_loss'):
                    self.reg_loss = tf.reduce_mean(reg_loss_list)
            else:
                self.reg_loss = 0.0

            with tf.variable_scope('cross_entropy_loss'):
                self.xent_loss = tf.losses.sparse_softmax_cross_entropy(target, output_layer)
            self.loss =  self.xent_loss + self.reg_loss

        with tf.variable_scope('metrics'):

            if num_target_class == 2:
                softmax = tf.nn.softmax(output_layer)
                self.prob_1 = tf.gather(softmax, 1, axis=1)
                self.auc, _auc_update_op = tf.metrics.auc(target, self.prob_1,
                                                          updates_collections='auc_update')

            prediction = tf.cast(tf.argmax(output_layer, 1), tf.int32)
            correct_pred = tf.equal(target, prediction)

            if num_target_class == 2:
                with tf.control_dependencies([_auc_update_op]):
                    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            else:
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss,
                                                              global_step=self.global_step)

        tf.summary.histogram('output_layer', output_layer)
        tf.summary.scalar('xent_loss', self.xent_loss)
        if reg_loss:
            tf.summary.scalar('reg_loss', self.reg_loss)
        # tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        if num_target_class == 2:
            with tf.device('cpu:0'):
                tf.summary.scalar('auc', self.auc)


