import numpy as np
import os
import logging
from ops import *
import tensorflow as tf

class Beta_VAE_Discrim():
    def __init__(self, encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
                 is_training, batch_size, input_data, final_act_fn, beta_s, beta_c,
                 output_prob,
                 learning_rate, optimizer, num_partition, discriminator_layer,
                 context_class_num, style_included, discrim_lambda):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.is_training = is_training
        self.batch_size = batch_size
        self.input_data = input_data['img']
        self.latent_class = tf.cast(tf.squeeze(tf.slice(input_data['latent'], [0, 0], [-1,
                                                                                    1])), tf.int32) - 1

        self._build_encoder(encoder_layer, fc_dim)

        self._build_latent(fc_dim, z_dim, num_partition, discriminator_layer,
                           context_class_num, style_included)

        self._build_decoder(decoder_layer, beta_s, fc_dim, final_act_fn, input_dim)

        self._build_recon_threshold(output_prob)
        self._build_loss_optimzier(output_prob, beta_s, beta_c, learning_rate,
                                   optimizer, discrim_lambda)
        self._build_summary(z_dim, beta_s)

    def _build_encoder(self, encoder_layer, fc_dim):
        pass

    def _build_latent(self, fc_dim, z_dim, num_partition, discriminator_layer,
                      class_num, style_included):

        with tf.variable_scope('latent'):
            # with tf.variable_scope('partitioned_layer'):
            #     partitioned_layer = make_partitioned_layer(self.encoder_output,
            #                                                num_partition,
            #                                                z_dim)
            #     if num_partition == 2:
            #         context = partitioned_layer[0]
            #         style = partitioned_layer[1]
            #     else:
            #         raise NotImplementedError

            self.z_m_c, self.z_log_sigma_sq_c, self.z_v_c = get_param(self.encoder_output, z_dim)
            self.z_m_s, self.z_log_sigma_sq_s, self.z_v_s = get_param(self.encoder_output, z_dim)

            self.latent = tf.concat([self.z_v_c, self.z_v_s], axis=1)
            self.permutation = tf.placeholder(tf.int32, [self.batch_size],
                                              name='permutation')
            

            if style_included:
                self.shuffled_style = tf.gather(self.z_m_s, self.permutation)
                self.discrim_input = tf.concat([self.z_m_c, self.shuffled_style], axis=1)
            else:
                self.discrim_input = self.z_m_c

            self._build_discriminator(self.discrim_input,
                                                           discriminator_layer, class_num)
            # self.discrim_logit_s = self._build_discriminator(self.z_v_s,
            #                                                  discriminator_layer,
            #                                                  class_num)

    def _build_discriminator(self, layer_input, discriminator_layer, context_class_num):

        with tf.variable_scope('discriminator'):
            discrim_layer_list = list()
            discrim_layer_list.append(layer_input)

            for layer_num, layer_config in enumerate(discriminator_layer):
                hidden_dim = layer_config
                with tf.variable_scope('layer_{}'.format(layer_num)):
                    layer_inputs = discrim_layer_list[-1]
                    if layer_num == len(discriminator_layer) - 1:
                        discrim_layer_list.append(tf.contrib.layers.fully_connected(
                            layer_inputs, context_class_num, None))
                    else:
                        discrim_layer_list.append(tf.contrib.layers.fully_connected(
                            layer_inputs, hidden_dim))
                    print('discrim : ', discrim_layer_list[-1].shape)
            self.discrim_logit = discrim_layer_list[-1]


    def _build_decoder(self, decoder_layer, beta_s, fc_dim, final_act_fn, input_dim):
        pass


    def _build_recon_threshold(self, output_prob):
        if output_prob == 'bernoulli':
            self.recon = tf.sigmoid(self.logit)
        elif output_prob == 'gaussian':
            self.recon = self.logit


    def _build_loss_optimzier(self, output_prob, beta_s, beta_c, learning_rate,
                              optimizer, discrim_lambda):
        with tf.variable_scope('loss'):
            if output_prob == 'bernoulli':
                self.recon_loss = tf.reduce_mean(tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.input_data,logits=self.logit), [1, 2, 3]))

            elif output_prob == 'gaussian':
                self.recon_loss = 0.5 * \
                                  tf.reduce_mean(
                                      tf.reduce_sum(tf.squared_difference(
                                          self.input_data, self.logit), [1,2,3]))

                ### 이거 loss 확인하기
            if beta_s != 0:
                with tf.variable_scope('style_latent_loss'):
                    latent_loss_raw_c = -0.5  * (1 + self.z_log_sigma_sq_c
                                                       - tf.square(self.z_m_c)
                                                       - tf.exp(self.z_log_sigma_sq_c))

                    self.latent_loss_z_c = tf.reduce_mean(latent_loss_raw_c, axis=0)
                    self.latent_loss_sliced_c = tf.unstack(self.latent_loss_z_c)

                    self.latent_loss_c = tf.reduce_mean(tf.reduce_sum(latent_loss_raw_c, axis=1))

                with tf.variable_scope('context_latent_loss'):
                    latent_loss_raw_s = -0.5 * (1 + self.z_log_sigma_sq_s
                                                - tf.square(self.z_m_s)
                                                - tf.exp(self.z_log_sigma_sq_s))

                    self.latent_loss_z_s = tf.reduce_mean(latent_loss_raw_s, axis=0)
                    self.latent_loss_sliced_s = tf.unstack(self.latent_loss_z_s)

                    self.latent_loss_s = tf.reduce_mean(
                        tf.reduce_sum(latent_loss_raw_s, axis=1))



                self.vae_loss = self.recon_loss + self.latent_loss_s * beta_s + \
                                self.latent_loss_c * beta_c

                with tf.variable_scope('discrim_loss'):
                    self.discrim_loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=self.latent_class, logits=self.discrim_logit)) * \
                                        discrim_lambda

                self.total_loss = self.vae_loss + self.discrim_loss



            # else:
            #     self.vae_loss = self.recon_loss
            #     self.latent_loss = tf.constant(0)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.contrib.layers.optimize_loss(self.total_loss,
                                                          self.global_step, learning_rate,
                                                            optimizer)

            self.train_op_vae = tf.contrib.layers.optimize_loss(self.vae_loss,
                                                            self.global_step,
                                                            learning_rate,
                                                            optimizer)

            self.train_op_discrim = tf.contrib.layers.optimize_loss(self.discrim_loss,
                                                                self.global_step,
                                                                learning_rate,
                                                                optimizer)

    def _build_summary(self, z_dim, beta_s):
        summary_list = list()

        self.predict = tf.cast(tf.argmax(self.discrim_logit, axis=1), tf.int32)



        recon_mean, recon_update = tf.contrib.metrics.streaming_mean(self.recon_loss,
                                                                     metrics_collections='streaming_loss',
                                                                     updates_collections='loss_update',
                                                                     name='recon')
        vae_mean, vae_update = tf.contrib.metrics.streaming_mean(self.recon_loss,
                                                                 metrics_collections='streaming_loss',
                                                                 updates_collections='loss_update',
                                                                 name='vae')
        latent_c_mean, latent_s_update = tf.contrib.metrics.streaming_mean(
            self.latent_loss_c,
                                                                       metrics_collections='streaming_loss',
                                                                       updates_collections='loss_update',
        name='latent_c')
        latent_c_mean, latent_s_update = tf.contrib.metrics.streaming_mean(
            self.latent_loss_s,
                                                                       metrics_collections='streaming_loss',
                                                                       updates_collections='loss_update',
        name='latent_s')

        discrim_mean, discrim_update = tf.metrics.accuracy(self.latent_class,
                                                           self.predict,
                                                           metrics_collections='streaming_loss',
                                                           updates_collections=
                                                           'loss_update',
                                                           name='discrim')

        self.loss_values = tf.placeholder(tf.float32, [5])
        recon_mean_val, vae_mean_val, latent_c_mean_val, latent_s_mean_val, \
        discrim_mean_val = tf.unstack(
            self.loss_values)
        summary_list.append(tf.summary.scalar('vae_loss', vae_mean_val))
        summary_list.append(tf.summary.scalar('recon_loss', recon_mean_val))
        summary_list.append(tf.summary.scalar('latent_c_loss', latent_c_mean_val))
        summary_list.append(tf.summary.scalar('latent_s_loss', latent_s_mean_val))
        summary_list.append(tf.summary.scalar('discrim_acc', discrim_mean_val))

        self.loss_summaries = tf.summary.merge(summary_list)

        summary_image_list = list()

        def __make_latent_summary(z_m, z_log_sigma_sq, latent_loss, latent_loss_sliced,
                                type):
            z_m_mean = tf.reduce_mean(z_m, axis=0)
            z_s_mean = tf.reduce_mean(z_log_sigma_sq, axis=0)
            z_m_sliced = tf.unstack(z_m_mean)
            z_s_sliced = tf.unstack(z_s_mean)
            for i in range(z_dim):
                summary_image_list.append(tf.summary.scalar('z_m_{}_{}'.format(type, i),
                                                            z_m_sliced[i]))
            for i in range(z_dim):
                summary_image_list.append(
                    tf.summary.scalar('z_{}_{}'.format(type, i), z_s_sliced[i]))


            summary_image_list.append(tf.summary.scalar('latent_loss_{}'.format(type),
                                                 latent_loss))
            for i in range(z_dim):
                summary_image_list.append(tf.summary.scalar('latent_loss_{}_{}'.format(
                    type, i), latent_loss_sliced[i]))

        __make_latent_summary(self.z_m_s, self.z_log_sigma_sq_s, self.latent_loss_s,
                              self.latent_loss_sliced_s,'s')
        __make_latent_summary(self.z_m_c, self.z_log_sigma_sq_c, self.latent_loss_c,
                              self.latent_loss_sliced_c, 'c')

        summary_image_list.append(tf.summary.image('original', self.input_data,
                                                   max_outputs=5))
        summary_image_list.append(tf.summary.image('recon', self.recon, max_outputs=5))


        self.image_summaries = tf.summary.merge(summary_image_list)



class Disentagled_VAE_FC_Discrim(Beta_VAE_Discrim):

    def __init__(self, encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
                 is_training, batch_size, input_data, final_act_fn, beta_s, beta_c,
                 output_prob,
                 learning_rate, optimizer, num_partition, discriminator_layer,
                 context_class_num, style_included, discrim_lambda):
        super().__init__(encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
                 is_training, batch_size, input_data, final_act_fn, beta_s, beta_c,
                 output_prob,
                 learning_rate, optimizer, num_partition, discriminator_layer,
                 context_class_num, style_included, discrim_lambda)

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

    def _build_decoder(self, decoder_layer, beta_s, fc_dim, final_act_fn, input_dim):

        with tf.variable_scope('decoder'):
            if beta_s != 0:
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


class Disentagled_VAE_CNN_Discrim(Beta_VAE_Discrim):
    pass

