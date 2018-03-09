import itertools
import os
import logging

import tensorflow as tf

import discriminator

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


class Disentagled_VAE():
    def __init__(self, encoder_layer, decoder_layer,
                 num_partition, num_output,
                 discriminator_layer,
                 is_training, is_bn, disentagled_act_fn, input_data, target_data):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.input_data = input_data
        self.target_data = target_data

        def fully_connected_bn(layer_input, num_outputs, activation_fn=tf.nn.relu):
            bn_layer = tf.contrib.layers.batch_norm(layer_input, is_training=is_training,
                                                    fused=True, center=True, scale=True)
            layer = tf.contrib.layers.fully_connected(bn_layer, num_outputs, activation_fn,
                                                      variables_collections=['W_b'])

            '''
            이런 방식으로 하는게 맞나..??? (is_training을 init에서 나온 scope에서 가져오독 하는것)
            '''
            return layer

        def fully_connected(layer_input, num_outputs, activation_fn=tf.nn.relu):
            layer = tf.contrib.layers.fully_connected(layer_input, num_outputs,
                                                      activation_fn,
                                                      variables_collections = ['W_b'])
            return layer

        def make_partitioned_layer(layer_input, num_partition, num_outputs, is_bn,
                                   activation_fn):
            partitioned_layer_list = list()
            for _ in range(num_partition):
                if is_bn:
                    partitioned_layer_list.append(fully_connected_bn(layer_input,
                                                                     num_outputs, activation_fn))
                else:
                    partitioned_layer_list.append(
                        fully_connected(layer_input, num_outputs, activation_fn))
            return partitioned_layer_list

        def layer_attention(layer_list):
            layer_output = list()
            for layer in layer_list:
                layer_dim = layer.shape.as_list()[1]
                bottle_neck_dim = int(layer_dim / 8)
                hidden = tf.contrib.layers.fully_connected(layer, bottle_neck_dim)
                hidden_2 = tf.contrib.layers.fully_connected(hidden, bottle_neck_dim)
                attention = tf.contrib.layers.fully_connected(hidden, 1,
                                                            activation_fn=tf.sigmoid)
                # attention = tf.contrib.layers.fully_connected(tmp, 1,
                #                                               activation=tf.sigmoid)
                # layer_output += layer * attention
                layer_output.append(layer * attention)
            layer_output = tf.concat(layer_output, axis=1)
            return layer_output

        with tf.variable_scope('encoder'):
            encoder_layer_list = list()
            encoder_layer_list.append(input_data)
            for layer_num, layer_dim in enumerate(encoder_layer):
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = encoder_layer_list[-1]
                    encoder_layer_list.append(fully_connected_bn(layer_inputs, layer_dim))


        with tf.variable_scope('disentagled_layer'):
            layer_inputs = encoder_layer_list[-1]
            part_layer_list = make_partitioned_layer(layer_inputs,
                                                    num_partition,
                                                num_output, is_bn, disentagled_act_fn)

            layer_discriminator = discriminator.Discriminator(part_layer_list, discriminator_layer)
            self.encoder_layer_output = tf.concat(part_layer_list, axis=1)


        # with tf.variable_scope('regularization_loss'):
        #     target_data


        with tf.variable_scope('decoder'):
            decoder_layer_list = list()
            decoder_layer_list.append(self.encoder_layer_output)
            for layer_num, layer_dim in enumerate(decoder_layer):
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = decoder_layer_list[-1]
                    decoder_layer_list.append(fully_connected_bn(layer_inputs,
                                                                 layer_dim))

            self.decoder_layer_output = fully_connected_bn(decoder_layer_list[-1],
                input_data.get_shape().as_list()[1], tf.sigmoid)


            with tf.variable_scope('loss'):
                # self.recon_loss = tf.losses.sigmoid_cross_entropy(input_data,
                #                                               decoder_layer_output)

                self.recon_loss = tf.losses.sigmoid_cross_entropy(input_data,
                                                               self.decoder_layer_output)
                self.discriminator_loss = layer_discriminator.loss
                # self.loss = self.recon_loss + self.discriminator_loss
                # self.dircrim_output = layer_discriminator.output
                # self.discrim_target = layer_discriminator.target
                # self.supervised_loss =

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op_discrim = tf.train.AdamOptimizer().minimize(
                self.discriminator_loss, global_step=self.global_step)
            self.train_op_gen = tf.train.AdamOptimizer().minimize(self.recon_loss,
                                                                  global_step=self.global_step)
            # self.train_op = tf.train.AdamOptimizer().minimize(self.loss,
            #                                                   global_step=self.global_step)

        tf.summary.scalar('recon_loss', self.recon_loss)
        tf.summary.scalar('dis_loss', self.discriminator_loss)


        with tf.variable_scope('embedding'):
            self.disentagled_layer_list = part_layer_list

            self.disentagled_layer_embedding = tf.convert_to_tensor(part_layer_list,
                                                               dtype=tf.float32)

            # self.embedding_var_1 = part_layer_list[0]
            # self.embedding_var_2 = part_layer_list[1]

            # self.embedding_var = self.disentagled_layer_list


        with tf.variable_scope('image_recon'):
            tf.summary.image('original', tf.reshape(input_data, (-1, 28, 28, 1)),
                                                    max_outputs=32)
            tf.summary.image('recon', tf.reshape(self.decoder_layer_output, (-1, 28, 28,
                                                                             1)),
                                                    max_outputs=32)

        # with tf.variable_scope('2d-plot'):
        #     for idx, disentagled_layer in enumerate(part_layer_list):
        #         i =  tf.linspace(0, 5, 10)
        #         j = tf.linspace(0, 5, 10)
        #         ii, jj = tf.meshgrid(i, j)
        #         range_input = tf.stack((ii, jj), axis=2).reshape(-1, 2)
        #         repeated = tf.tile(disentagled_layer, range_input.get_shape().as_list[0])
        #         if idx == 0:
        #             tf.concat([range_input, repeated], 1)
        #



