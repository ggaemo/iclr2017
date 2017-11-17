
import itertools
import numpy as np
import os
import logging
from ops import *
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

class Beta_VAE():
    def __init__(self, encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
                 is_training, batch_size, input_data, final_act_fn, beta, output_prob,
                 learning_rate, optimizer):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.is_training = is_training
        self.input_data = input_data['img']

        self._build_encoder(encoder_layer, fc_dim)
        if beta != 0:
            self._build_latent(fc_dim, z_dim)
        self._build_decoder(decoder_layer, beta, fc_dim, final_act_fn, input_dim)

        self._build_recon_threshold(output_prob)
        self._build_loss_optimzier(output_prob, beta, learning_rate, optimizer)
        self._build_summary(z_dim, beta)

    def _build_encoder(self, encoder_layer, fc_dim):
        pass

    def _build_latent(self, fc_dim, z_dim):
        with tf.variable_scope('latent'):
            z_m, z_log_sigma_sq, z_v = get_param(self.encoder_output, z_dim)

            self.z_m = z_m
            self.z_log_sigma_sq = z_log_sigma_sq
            self.latent = z_v

    def _build_decoder(self, decoder_layer, beta, fc_dim, final_act_fn, input_dim):
        pass

    def _build_recon_threshold(self, output_prob):
        if output_prob == 'bernoulli':
            self.recon = tf.sigmoid(self.logit)
            self.threshold = tf.round(self.recon)
        elif output_prob == 'gaussian':
            self.recon = self.logit
            self.threshold = self.logit

    def _build_loss_optimzier(self, output_prob, beta, learning_rate, optimizer):
        with tf.variable_scope('loss'):
            if output_prob == 'bernoulli':
                self.recon_loss = tf.reduce_mean(tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.input_data,logits=self.logit), [1, 2, 3]))

                # decoder_output_dist = output_prob(decoder_layer_list[-1])
                # decoder_output_mean = decoder_output_dist.probs

                # self.recon_loss = -tf.reduce_mean(
                #     tf.reduce_sum(self.input_data * tf.log(1e-10 +self.recon)
                #                   +(1-self.input_data) * tf.log(1e-10+1-self.recon), 1))
            elif output_prob == 'gaussian':
                self.recon_loss = 0.5 * \
                                  tf.reduce_mean(
                                      tf.reduce_sum(tf.squared_difference(
                                          self.input_data, self.logit), [1,2,3]))

                ### 이거 loss 확인하기
            if beta != 0:
                latent_loss_raw = -0.5  * (1 + self.z_log_sigma_sq
                                                   - tf.square(self.z_m)
                                                   - tf.exp(self.z_log_sigma_sq))

                self.latent_loss_z = tf.reduce_mean(latent_loss_raw, axis=0)
                self.latent_loss_sliced = tf.unstack(self.latent_loss_z)

                self.latent_loss = tf.reduce_mean(tf.reduce_sum(latent_loss_raw, axis=1))

                self.vae_loss = self.recon_loss + self.latent_loss * beta
            else:
                self.vae_loss = self.recon_loss
                self.latent_loss = tf.constant(0)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.contrib.layers.optimize_loss(self.vae_loss,
                                                          self.global_step, learning_rate,
                                                            optimizer)

    def _build_summary(self, z_dim, beta):
        summary_list = list()

        recon_mean, recon_update = tf.contrib.metrics.streaming_mean(self.recon_loss,
                                                                     metrics_collections='streaming_loss',
                                                                     updates_collections='loss_update')
        vae_mean, vae_update = tf.contrib.metrics.streaming_mean(self.recon_loss,
                                                                 metrics_collections='streaming_loss',
                                                                 updates_collections='loss_update')
        latent_mean, latent_update = tf.contrib.metrics.streaming_mean(self.latent_loss,
                                                                       metrics_collections='streaming_loss',
                                                                       updates_collections='loss_update')

        self.loss_values = tf.placeholder(tf.float32, [3])
        recon_mean_val, vae_mean_val, latent_mean_val = tf.unstack(self.loss_values)
        summary_list.append(tf.summary.scalar('vae_loss', vae_mean_val))
        summary_list.append(tf.summary.scalar('recon_loss', recon_mean_val))
        summary_list.append(tf.summary.scalar('latent_loss', latent_mean_val))

        self.loss_summaries = tf.summary.merge(summary_list)

        summary_image_list = list()

        if beta != 0:
            z_m_mean = tf.reduce_mean(self.z_m, axis=0)
            z_s_mean = tf.reduce_mean(self.z_log_sigma_sq, axis=0)
            z_m_sliced = tf.unstack(z_m_mean)
            z_s_sliced = tf.unstack(z_s_mean)
            for i in range(z_dim):
                summary_image_list.append(tf.summary.scalar('z_m_{}'.format(i), z_m_sliced[
                    i]))

            for i in range(z_dim):
                summary_image_list.append(
                    tf.summary.scalar('z_s_{}'.format(i), z_s_sliced[i]))
            tf.summary.scalar('latent_loss', self.latent_loss)
            for i in range(z_dim):
                summary_image_list.append(tf.summary.scalar('latent_loss_{}'.format(i),
                                                            self.latent_loss_sliced[i]))

        summary_image_list.append(tf.summary.image('original', self.input_data,
                                                   max_outputs=5))
        summary_image_list.append(tf.summary.image('recon', self.recon, max_outputs=5))
        summary_image_list.append(tf.summary.image('threshold', self.threshold,
                                                   max_outputs=5))

        self.image_summaries = tf.summary.merge(summary_image_list)


class Disentagled_VAE_CNN(Beta_VAE):
    def __init__(self, encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
                 is_training, batch_size, input_data, final_act_fn, beta, output_prob,
                 learning_rate, optimizer):
        reduced_img_size = int(np.ceil(input_dim / np.prod([i[2]  for i in
                                                            encoder_layer])))
        reduced_layer_size =  reduced_img_size ** 2
        self.encoder_layer_last_shape = (-1, reduced_img_size, reduced_img_size, encoder_layer[-1][0])
        self.encoder_flattened_dim = int( reduced_layer_size * encoder_layer[-1][0])
        self.padding = 'SAME'

        super().__init__(encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
                 is_training, batch_size, input_data, final_act_fn, beta, output_prob,
                 learning_rate, optimizer)


    def _build_encoder(self, encoder_layer, fc_dim):
        with tf.variable_scope('encoder'):
            encoder_layer_list = list()
            encoder_layer_list.append(self.input_data)
            for layer_num, layer_config in enumerate(encoder_layer):
                filters, kernel_size, stride = layer_config
                kernel_size = (kernel_size, kernel_size)
                strides = (stride, stride)
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = encoder_layer_list[-1]
                    encoder_layer_list.append(conv_block_encode(layer_inputs, filters,
                                                         kernel_size, strides,
                                                                self.padding,
                                                         self.is_training))
                    print('enc :',encoder_layer_list[-1].shape)

            encoder_output = tf.contrib.layers.flatten(encoder_layer_list[-1])

            # self.encoder_last_shape = [-1] + encoder_layer_list[-1].shape.as_list()[1:]
            self.encoder_output = tf.contrib.layers.fully_connected(encoder_output,
                                                             fc_dim)

            # encoder_output = tf.reshape(encoder_layer_list[-1], [batch_size, -1])


    def _build_decoder(self, decoder_layer, beta, fc_dim, final_act_fn, input_dim):

        with tf.variable_scope('decoder'):

            if beta != 0:
                decoder_input = tf.contrib.layers.fully_connected(self.latent, fc_dim)
            else:
                decoder_input = tf.contrib.layers.fully_connected(self.encoder_output,
                                                                  fc_dim)


            decoder_input = tf.contrib.layers.fully_connected(decoder_input,
                                                              self.encoder_flattened_dim)
            self.decoder_input = tf.reshape(decoder_input, self.encoder_layer_last_shape)

            print('reshaped_dim', self.encoder_layer_last_shape)

            decoder_layer_list = list()

            decoder_layer_list.append(self.decoder_input)

            for layer_num, layer_config in enumerate(decoder_layer):
                filters, kernel_size, stride = layer_config
                kernel_size = (kernel_size, kernel_size)
                strides = (stride, stride)
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = decoder_layer_list[-1]
                    if layer_num == len(decoder_layer) - 1:
                        decoder_layer_list.append(conv_block_decode(layer_inputs, filters,
                                                                    kernel_size, strides,
                                                                    self.padding,
                                                                    self.is_training,
                                                                    final_act_fn))
                    else:
                        decoder_layer_list.append(conv_block_decode(layer_inputs, filters,
                                                                    kernel_size, strides,
                                                                    self.padding,
                                                                    self.is_training))
                    print('dec : ', decoder_layer_list[-1].shape)

            decoder_output = decoder_layer_list[-1]
            self.logit = decoder_output


class Disentagled_VAE_FC(Beta_VAE):

    def __init__(self, encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
                 is_training, batch_size, input_data, final_act_fn, beta, output_prob,
                 learning_rate, optimizer):
        super().__init__(encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
                 is_training, batch_size, input_data, final_act_fn, beta, output_prob,
                 learning_rate, optimizer)
        # self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # self.is_training = is_training
        # self.input_data = input_data['image']
        #
        # self._build_encoder(encoder_layer, fc_dim)
        # if beta != 0:
        #     self._build_latent(fc_dim, z_dim)
        # self._build_decoder(decoder_layer, beta, fc_dim, final_act_fn, input_dim)
        # self._build_recon_threshold(output_prob)
        # self._build_summary(z_dim, beta)


    def _build_encoder(self, encoder_layer, fc_dim):
        with tf.variable_scope('encoder'):
            encoder_layer_list = list()
            encoder_layer_list.append(tf.contrib.layers.flatten(self.input_data))
            for layer_num, layer_config in enumerate(encoder_layer):
                hidden_dim = layer_config
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = encoder_layer_list[-1]
                    encoder_layer_list.append(tf.contrib.layers.fully_connected(
                        layer_inputs, hidden_dim))
                    print('enc : ', encoder_layer_list[-1].shape)
            encoder_output = encoder_layer_list[-1]
            self.encoder_output = tf.contrib.layers.fully_connected(encoder_output,
                                                             fc_dim)

    def _build_decoder(self, decoder_layer, beta, fc_dim, final_act_fn, input_dim):

        with tf.variable_scope('decoder'):
            if beta != 0:
                self.decoder_input = tf.contrib.layers.fully_connected(self.latent,
                                                                       fc_dim)
            else:
                self.decoder_input = tf.contrib.layers.fully_connected(self.encoder_output,
                                                                       fc_dim)
            decoder_layer_list = list()
            decoder_layer_list.append(self.decoder_input)

            for layer_num, layer_config in enumerate(decoder_layer):
                hidden_dim = layer_config
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = decoder_layer_list[-1]
                    if layer_num == len(decoder_layer)-1:
                        decoder_layer_list.append(tf.contrib.layers.fully_connected(
                            layer_inputs, hidden_dim, final_act_fn))
                    else:
                        decoder_layer_list.append(tf.contrib.layers.fully_connected(
                            layer_inputs, hidden_dim, tf.nn.tanh))
                    print('dec : ', decoder_layer_list[-1].shape)

                    """tf.contrib.layers의 conv2d를 쓸지, tf.nn.conv2d를 쓸지
                    """
            decoder_output = decoder_layer_list[-1]

            # decoder_output_dist = output_prob(logits=decoder_output)
            # decoder_output_mean = decoder_output_dist.probs
            # decoder_output_logits = decoder_output_dist.logits
            # decoder_output_mean = decoder_output

            self.logit = tf.reshape(decoder_output, (-1, input_dim,
                                                                 input_dim, 1))