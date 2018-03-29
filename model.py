
from ops import *
import tensorflow as tf


class Beta_VAE_Discrim():
    def __init__(self, encoder_layer, decoder_layer, input_dim, z_dim,
                 is_training, batch_size,  beta_s, beta_c,
                 output_prob,
                 learning_rate, optimizer, discriminator_layer, classifier_layer,
                 context_class_num, discrim_lambda, classifier_lambda, deterministic_c,lambda_d_reg):

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.is_training = is_training
        self.batch_size = batch_size
        self.context_class_num = context_class_num
        self.input_data = tf.placeholder(tf.float32, [None, input_dim, input_dim, 1])
        self.latent = tf.placeholder(tf.int64, [None, 1])

        self.latent_class = self.latent
        self.latent_class_onehot =tf.one_hot(self.latent_class, context_class_num)


        with tf.variable_scope('encoder'):
            self.encoder_output, self.encoder_output_shape = self._build_encoder(
                encoder_layer)

        with tf.variable_scope('latent'):
            self.latent_c_copied, self.latent_s_copied, self.latent_s_random = \
                self._build_latent(self.encoder_output, z_dim, discriminator_layer,
                                   context_class_num, deterministic_c)

        with tf.variable_scope('decoder'):
            self.recon_logit_c_copied = self._build_decoder(self.latent_c_copied,
                                                            decoder_layer,
                                                            beta_s,
                                                            input_dim, self.encoder_output_shape)
        with tf.variable_scope('decoder', reuse=True):
            self.recon_logit_s_copied = self._build_decoder(self.latent_s_copied,
                                                            decoder_layer, beta_s,
                                                            input_dim, self.encoder_output_shape)
        with tf.variable_scope('decoder', reuse=True):
            self.recon_logit_s_copied = self._build_decoder(self.latent_s_copied,
                                                            decoder_layer, beta_s,
                                                            input_dim, self.encoder_output_shape)
        with tf.variable_scope('decoder', reuse=True):
            self.recon_logit_s_random = self._build_decoder(self.latent_s_random,
                                                            decoder_layer, beta_s,
                                                            input_dim, self.encoder_output_shape)

        self.recon_c_copied = self._build_recon_threshold(self.recon_logit_c_copied, output_prob)

        self.recon_s_copied = self._build_recon_threshold(self.recon_logit_s_copied,
                                                          output_prob)

        self.recon_s_random =  self._build_recon_threshold(self.recon_logit_s_random,
                                                          output_prob)


        epsilon = tf.random_uniform((tf.shape(self.recon_s_random)[0], 1, 1, 1), 0.0, 1.0)

        self.recon_grad_penalty_s_random = epsilon * self.input_data + (1 - epsilon) * \
                                                         self.recon_s_random
        self.recon_grad_penalty_s_recon = epsilon * self.input_data + (1 - epsilon) * \
                                                               self.recon_s_copied

        self.recon_grad_penalty = self.recon_grad_penalty_s_recon + \
                                  self.recon_grad_penalty_s_random

        with tf.variable_scope('discriminator'):
            self.discrim_logit_of_input = self._build_discriminator(
                self.input_data, discriminator_layer, 1)

            self.discrim_input_acc = tf.reduce_mean(tf.sigmoid(
                self.discrim_logit_of_input))

        with tf.variable_scope('discriminator', reuse=True):
            self.discrim_logit_of_recon_s_copied = self._build_discriminator(
                self.recon_s_copied, discriminator_layer, 1)

            self.discrim_recon_s_acc = tf.reduce_mean(
                tf.sigmoid(self.discrim_logit_of_recon_s_copied))

        with tf.variable_scope('discriminator', reuse=True):
            self.discrim_logit_of_recon_s_random = self._build_discriminator(
                self.recon_s_random, discriminator_layer, 1)

            self.discrim_recon_s_random_acc = tf.reduce_mean(
                tf.sigmoid(self.discrim_logit_of_recon_s_random))

        with tf.variable_scope('discriminator', reuse=True):
            self.discrim_logit_of_recon_c_copied = self._build_discriminator(
                self.recon_c_copied, discriminator_layer, 1)

            self.discrim_recon_c_acc = tf.reduce_mean(
                tf.sigmoid(self.discrim_logit_of_recon_c_copied))

        with tf.variable_scope('discriminator', reuse=True):
            self.discrim_logit_of_recon_gp_s_recon = self._build_discriminator(
                self.recon_grad_penalty_s_recon, discriminator_layer, 1)

        with tf.variable_scope('discriminator', reuse=True):
            self.discrim_logit_of_gp_s_random = self._build_discriminator(
                self.recon_grad_penalty_s_random, discriminator_layer, 1)


        with tf.variable_scope('classifier'):
            self.class_logit_of_input = self._build_discriminator(
                self.input_data, classifier_layer, context_class_num)

            self.class_pred_input = tf.cast(tf.argmax(self.class_logit_of_input, 1),
            tf.int32)

        with tf.variable_scope('classifier', reuse=True):
            self.class_logit_of_recon_s_copied = self._build_discriminator(
                self.recon_s_copied, classifier_layer, context_class_num)

            self.class_pred_recon_s_copied = tf.cast(tf.argmax(self.class_logit_of_recon_s_copied,
                                                              1),tf.int32)

        with tf.variable_scope('classifier', reuse=True):
            self.class_logit_of_recon_s_random = self._build_discriminator(
                self.recon_s_random, classifier_layer, context_class_num)

            self.class_pred_recon_s_random = tf.cast(tf.argmax(
                self.class_logit_of_recon_s_random,
                                                              1),tf.int32)

        with tf.variable_scope('classifier', reuse=True):
            self.class_logit_of_recon_c_copied = self._build_discriminator(
                self.recon_c_copied, classifier_layer, context_class_num)

            self.class_pred_recon_c_copied = tf.cast(tf.argmax(
                self.class_logit_of_recon_c_copied, 1),tf.int32)

        self._build_loss_optimzier(output_prob, beta_s, beta_c, learning_rate,
                                   optimizer, discrim_lambda, classifier_lambda,
                                   self.z_m_c,
                              self.z_log_sigma_sq_c, self.z_m_s,
                              self.z_log_sigma_sq_s, self.recon_logit_c_copied,
                                   deterministic_c, lambda_d_reg)

        self._build_summary(z_dim, deterministic_c)

    def _build_encoder(self, encoder_layer):
        pass

    def _build_latent(self, encoder_output, z_dim, discriminator_layer,
                      class_num, deterministic_c):


        self.z_m_c, self.z_log_sigma_sq_c, self.z_v_c = get_param(encoder_output,
                                                                  z_dim,
                                                                  deterministic_c)

        self.z_m_s, self.z_log_sigma_sq_s, self.z_v_s = get_param(encoder_output,
                                                                  z_dim, False)
        self.permutation_s = tf.placeholder(tf.int32, [None],
                                          name='permutation_s')
        self.permutation_c = tf.placeholder(tf.int32, [None],
                                            name='within_class_copy')

        self.partitioned_c = tf.dynamic_partition(self.z_v_c, self.permutation_c,
                                                     class_num * 2)

        self.partitioned_c_copied = [self.partitioned_c[int(i / 2) * 2] for i in range(2
                                                                              *class_num)]

        self.z_v_c_copied = tf.concat(self.partitioned_c_copied, axis=0)

        latent_c_copied = tf.concat([self.z_v_c_copied, self.z_v_s], axis=1)

        # latent_c_copied = self.z_v_c_copied + self.z_v_s

        self.partitioned_s = tf.dynamic_partition(self.z_v_s, self.permutation_s,
                                                  class_num * 2)

        self.partitioned_s_copied = [self.partitioned_s[int(i / 2) * 2] for i in
                                     range(class_num * 2)]

        self.z_v_s_copied = tf.concat(self.partitioned_s_copied, axis=0)

        latent_s_copied = tf.concat([self.z_v_c_copied, self.z_v_s_copied],
                                           axis=1)
        #
        # latent_s_copied = self.z_v_c_copied + self.z_v_s_copied

        latent_s_random = tf.concat([self.z_v_c_copied,
                                     tf.random_normal(tf.shape(self.z_v_s_copied))
                                     ], axis=1)

        # latent_s_random = self.z_v_c_copied + tf.random_normal(tf.shape(self.z_v_s_copied))


        return latent_c_copied, latent_s_copied, latent_s_random

    def _build_discriminator(self, layer_input, discriminator_layer, class_num):
        pass

    def _build_decoder(self, layer_input, decoder_layer, beta_s, 
                       input_dim, encoder_output_shape):
        pass

    def _build_recon_threshold(self, logit, output_prob):
        if output_prob == 'bernoulli':
            recon = tf.sigmoid(logit)
        elif output_prob == 'gaussian':
            recon = logit
        return recon

    def _build_loss_optimzier(self, output_prob, beta_s, beta_c, learning_rate,
                              optimizer, discrim_lambda, classifier_lambda, z_m_c,
                              z_log_sigma_sq_c, z_m_s,
                              z_log_sigma_sq_s, recon_logit, determinsitic_c, lambda_d_reg):

        with tf.variable_scope('loss'):

            with tf.variable_scope('recon_loss'):
                if output_prob == 'bernoulli':
                    self.recon_loss = tf.reduce_mean(tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=self.input_data, logits=recon_logit), [1, 2, 3]))

                elif output_prob == 'gaussian':
                    self.recon_loss = 0.5 * \
                                      tf.reduce_mean(
                                          tf.reduce_sum(tf.squared_difference(
                                              self.input_data, recon_logit), [1, 2, 3]))

            with tf.variable_scope('context_latent_loss'):
                if determinsitic_c:
                    self.latent_loss_c = 0
                    self.latent_loss_sliced_c = 0
                    self.latent_loss_z_c = 0
                else:
                    latent_loss_raw_c = -0.5 * (1 + z_log_sigma_sq_c
                                                - tf.square(z_m_c)
                                                - tf.exp(z_log_sigma_sq_c))

                    self.latent_loss_z_c = tf.reduce_mean(latent_loss_raw_c, axis=0)
                    self.latent_loss_sliced_c = tf.unstack(self.latent_loss_z_c)

                    self.latent_loss_c = tf.reduce_mean(
                        tf.reduce_sum(latent_loss_raw_c, axis=1))


            with tf.variable_scope('style_latent_loss'):
                latent_loss_raw_s = -0.5 * (1 + z_log_sigma_sq_s
                                            - tf.square(z_m_s)
                                            - tf.exp(z_log_sigma_sq_s))

                self.latent_loss_z_s = tf.reduce_mean(latent_loss_raw_s, axis=0)
                self.latent_loss_sliced_s = tf.unstack(self.latent_loss_z_s)

                self.latent_loss_s = tf.reduce_mean(
                    tf.reduce_sum(latent_loss_raw_s, axis=1))

            with tf.variable_scope('discrim_loss'):

                self.discrim_loss_recon_s_gen = tf.reduce_mean(self.discrim_logit_of_recon_s_copied)
                self.discrim_loss_recon_s_rand_gen = tf.reduce_mean(self.discrim_logit_of_recon_s_random)
                self.discrim_loss_input_adv = tf.reduce_mean(self.discrim_logit_of_input)

                self.discrim_loss_gp = tf.reduce_mean((self.discrim_logit_of_gp_s_random,
                self.discrim_logit_of_recon_gp_s_recon))

                discrim_gradients_s_random = tf.gradients(
                    self.discrim_logit_of_gp_s_random,
                                                 self.recon_grad_penalty_s_random)[0]

                discrim_gradients_s_recon = tf.gradients(
                    self.discrim_logit_of_recon_gp_s_recon,
                                                 self.recon_grad_penalty_s_recon)[0]


                grad_norm_s_random = tf.sqrt(tf.reduce_sum(discrim_gradients_s_random **
                                                           2, axis=[1, 2]))
                grad_norm_s_recon = tf.sqrt(
                    tf.reduce_sum(discrim_gradients_s_recon ** 2, axis=[1, 2]))

                d_reg_s_random = tf.reduce_mean((grad_norm_s_random - 1) ** 2)
                d_reg_s_recon = tf.reduce_mean((grad_norm_s_recon - 1) ** 2)
                d_reg = tf.reduce_mean([d_reg_s_recon, d_reg_s_random])

            with tf.variable_scope('classifier_loss'):
                self.xent_loss_input = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=self.latent_class_onehot,
                        logits=self.class_logit_of_input
                    )
                )

                self.xent_loss_recon_s = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=self.latent_class_onehot,
                        logits=self.class_logit_of_recon_s_copied
                    ) * classifier_lambda
                )

                self.xent_loss_recon_s_rand = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=self.latent_class_onehot,
                        logits=self.class_logit_of_recon_s_random
                    ) * classifier_lambda
                )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        discrim_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='Model/discriminator')

        classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='Model/classifier')

        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='Model/encoder')

        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Model/decoder')

        latent_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Model/latent')

        vae_vars = list(set(tf.trainable_variables()) - set(discrim_vars) - set(classifier_vars))
        # vae_vars = encoder_vars + decoder_vars + latent_vars # same as above

        self.vae_loss = self.recon_loss + self.latent_loss_s * beta_s

        self.discrim_fake_loss = tf.reduce_mean( (self.discrim_loss_recon_s_gen \
                                        + self.discrim_loss_recon_s_rand_gen))

        self.discrim_real_loss = self.discrim_loss_input_adv

        self.discrim_w_loss = self.discrim_real_loss - self.discrim_fake_loss

        self.discrim_w_loss_gp = self.discrim_w_loss - lambda_d_reg * d_reg

        self.classifier_loss = self.xent_loss_input
        self.classifier_loss_gen = 0.5 * (self.xent_loss_recon_s
                                        +self.xent_loss_recon_s_rand)

        with tf.control_dependencies(update_ops):

            if optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(1e-4, 0.5, 0.9)

            self.train_op_vae = tf.contrib.layers.optimize_loss(self.vae_loss,
                                                                self.global_step,
                                                                learning_rate,
                                                                optimizer,
                                                                variables=vae_vars)

            self.train_op_w = tf.contrib.layers.optimize_loss(
                    -self.discrim_w_loss_gp,
                    self.global_step, learning_rate, optimizer, variables=discrim_vars)

            self.train_op_w_gen = tf.contrib.layers.optimize_loss(
                -self.discrim_fake_loss, self.global_step, learning_rate, optimizer,
            variables=vae_vars)

            self.train_op_classifier = tf.contrib.layers.optimize_loss(
                self.classifier_loss,
                self.global_step,learning_rate, optimizer,
                variables=classifier_vars
            )

            self.train_op_classifier_gen = tf.contrib.layers.optimize_loss(
                self.classifier_loss_gen,
                self.global_step, learning_rate, optimizer,
                variables=vae_vars
            )

    def _build_summary(self, z_dim, deterministic_c):
        summary_list = list()

        # self.predict = tf.cast(tf.argmax(self.discrim_logit, axis=1), tf.int32)

        recon_mean, recon_update = tf.contrib.metrics.streaming_mean(self.recon_loss,
                                                                     metrics_collections='streaming_loss',
                                                                     updates_collections='loss_update',
                                                                     name='recon')
        vae_mean, vae_update = tf.contrib.metrics.streaming_mean(self.vae_loss,
                                                                 metrics_collections='streaming_loss',
                                                                 updates_collections='loss_update',
                                                                 name='vae')

        latent_s_mean, latent_s_update = tf.contrib.metrics.streaming_mean(
            self.latent_loss_s,
            metrics_collections='streaming_loss',
            updates_collections='loss_update',
            name='latent_s')

        discrim_w_mean, discrim_w_update = tf.contrib.metrics.streaming_mean(
            self.discrim_w_loss,
            metrics_collections='streaming_loss',
            updates_collections='loss_update',
            name='discrim_w_loss')

        discrim_recon_s_acc_mean, discrim_recon_s_acc_update = \
            tf.contrib.metrics.streaming_mean(
            self.discrim_recon_s_acc,
            metrics_collections='streaming_loss',
            updates_collections='loss_update',
            name='discrim_recon_s_acc')

        discrim_recon_c_acc_mean, discrim_recon_c_acc_update = \
            tf.contrib.metrics.streaming_mean(
                self.discrim_recon_c_acc,
                metrics_collections='streaming_loss',
                updates_collections='loss_update',
                name='discrim_recon_c_acc')

        discrim_input_acc_mean, discrim_input_acc_update = \
            tf.contrib.metrics.streaming_mean(
            self.discrim_input_acc,
            metrics_collections='streaming_loss',
            updates_collections='loss_update',
            name='discrim_input_acc')

        class_input_xent_mean, class_input_xent_update= tf.contrib.metrics.streaming_mean(
            self.xent_loss_input,
            metrics_collections='streaming_loss',
            updates_collections='loss_update',
            name='class_input_loss')

        class_recon_s_xent_mean, class_recon_s_xent_update = \
            tf.contrib.metrics.streaming_mean(
            self.xent_loss_recon_s,
            metrics_collections='streaming_loss',
            updates_collections='loss_update',
            name='class_recon_s_loss')

        class_input_acc, class_input_acc_update = tf.metrics.accuracy(
            self.latent_class,
            self.class_pred_input,
            metrics_collections='streaming_loss',
            updates_collections='loss_update',
            name='class_input_acc')

        class_recon_s_acc, class_recon_s_acc_update = \
            tf.metrics.accuracy(
                self.latent_class,
            self.class_pred_recon_s_copied,
            metrics_collections='streaming_loss',
            updates_collections='loss_update',
            name='class_recon_s_acc')

        class_recon_c_acc, class_recon_c_acc_update = \
            tf.metrics.accuracy(
                self.latent_class,
                self.class_pred_recon_c_copied,
                metrics_collections='streaming_loss',
                updates_collections='loss_update',
                name='class_recon_c_acc')

        self.loss_values = tf.placeholder(tf.float32, [12])
        recon_mean_val, vae_mean_val, latent_s_mean_val, \
        discrim_w_mean_val, \
        discrim_recon_s_acc_val, discrim_recon_c_acc_val, discrim_input_acc_val,  \
        class_input_loss_val, \
        class_recon_s_loss_val, class_input_acc_val, class_recon_s_acc_val, \
        class_recon_c_acc_val \
         = tf.unstack(self.loss_values)
        summary_list.append(tf.summary.scalar('recon_loss', recon_mean_val))
        summary_list.append(tf.summary.scalar('vae_loss', vae_mean_val))
        summary_list.append(tf.summary.scalar('latent_s_loss', latent_s_mean_val))
        summary_list.append(tf.summary.scalar('discrim_w_loss', discrim_w_mean_val))
        summary_list.append(tf.summary.scalar('class_input_loss', class_input_loss_val))
        summary_list.append(tf.summary.scalar('class_recon_loss',
                                              class_recon_s_loss_val))
        summary_list.append(tf.summary.scalar('class_input_acc', class_input_acc_val))
        summary_list.append(tf.summary.scalar('class_recon_s_acc', class_recon_s_acc_val))
        summary_list.append(tf.summary.scalar('class_recon_c_acc', class_recon_c_acc_val))
        summary_list.append(tf.summary.scalar('discrim_input_acc', discrim_input_acc_val))
        summary_list.append(tf.summary.scalar('discrim_recon_s_acc',
                                              discrim_recon_s_acc_val))
        summary_list.append(tf.summary.scalar('discrim_recon_c_acc',
                                              discrim_recon_c_acc_val))


        self.loss_summaries = tf.summary.merge(summary_list)

        summary_image_list = list()

        num_per_class = int(self.batch_size / self.context_class_num)

        for i in range(self.context_class_num):
            prev_idx = i *num_per_class
            next_idx = (i+1) * num_per_class
            summary_image_list.append(
                tf.summary.image('original_{}'.format(i),self.input_data[prev_idx:next_idx],
                                                       max_outputs=3))
            summary_image_list.append(
                tf.summary.image('recon_c_copied_{}'.format(i), self.recon_c_copied[
                    prev_idx:next_idx],
                                 max_outputs=3))
            summary_image_list.append(tf.summary.image('recon_s_swapped_{}'.format(i),
                                                       self.recon_s_copied[
                                                       prev_idx:next_idx],
                                                       max_outputs=3))
            summary_image_list.append(tf.summary.image('recon_s_random_{}'.format(i),
                                                       self.recon_s_random[
                                                           prev_idx:next_idx],
                                                       max_outputs=3))

        self.image_summaries = tf.summary.merge(summary_image_list)


class Disentagled_VAE_FC_Discrim(Beta_VAE_Discrim):
    def __init__(self, encoder_layer, decoder_layer, input_dim, z_dim,
                 is_training, batch_size,  beta_s, beta_c,
                 output_prob,
                 learning_rate, optimizer, discriminator_layer, classifier_layer,
                 context_class_num, discrim_lambda, classifier_lambda, deterministic_c, lambda_d_reg):
        super().__init__(encoder_layer, decoder_layer, input_dim, z_dim,
                         is_training, batch_size,  beta_s, beta_c,
                         output_prob,
                         learning_rate, optimizer, discriminator_layer,classifier_layer,
                         context_class_num, discrim_lambda, classifier_lambda,
                         deterministic_c, lambda_d_reg)

    def _build_encoder(self, encoder_layer):
        encoder_layer_list = list()
        encoder_layer_list.append(tf.contrib.layers.flatten(self.input_data))

        for layer_num, layer_config in enumerate(encoder_layer):
            hidden_dim = layer_config
            if layer_num ==len(encoder_layer) - 1:
                normalizer_fn = None
                normalizer_params = None
            else:
                normalizer_fn = tf.contrib.layers.batch_norm
                normalizer_params = {'is_training'
                                     : self.is_training
                                     }

            with tf.variable_scope('layer_{}'.format(layer_num)):
                layer_inputs = encoder_layer_list[-1]
                layer_output = tf.contrib.layers.fully_connected(
                    layer_inputs, hidden_dim, lambda x: tf.nn.leaky_relu(x, alpha=0.2),
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params
                )
                encoder_layer_list.append(layer_output)

                print('enc : ', encoder_layer_list[-1].shape)

        encoder_output_shape = encoder_layer_list[-1].shape.shape.as_list()[1]

        encoder_output = encoder_layer_list[-1]
        return encoder_output, encoder_output_shape

    def _build_decoder(self, layer_input, decoder_layer, beta_s,  
                       input_dim, encoder_output_shape):

        decoder_layer_list = list()
        decoder_layer_list.append(layer_input)

        for layer_num, layer_config in enumerate(decoder_layer):
            hidden_dim = layer_config
            with tf.variable_scope('layer_{}'.format(layer_num)):
                layer_inputs = decoder_layer_list[-1]
                if layer_num == len(decoder_layer) - 1:
                    normalizer_fn = None
                    normalizer_params = None
                    activation_fn = tf.identity

                else:
                    normalizer_fn = tf.contrib.layers.batch_norm
                    normalizer_params = {'is_training': self.is_training}
                    activation_fn = lambda x: tf.nn.leaky_relu(x,alpha=0.2)


                layer_output = tf.contrib.layers.fully_connected(
                    layer_inputs, hidden_dim, activation_fn,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params
                )

                decoder_layer_list.append(layer_output)

                print('dec : ', decoder_layer_list[-1].shape)

        decoder_output = decoder_layer_list[-1]

        logit = tf.reshape(decoder_output, (-1, input_dim,
                                                 input_dim, 1))
        return logit

    def _build_discriminator(self, layer_input, discriminator_layer, class_num):
        discrim_layer_list = list()
        discrim_layer_list.append(layer_input)

        for layer_num, layer_config in enumerate(discriminator_layer):

            with tf.variable_scope('layer_{}'.format(layer_num)):
                layer_inputs = discrim_layer_list[-1]

                if layer_num == 0:
                    normalizer_fn = None
                    normalizer_params = None
                else:
                    normalizer_fn = tf.contrib.layers.batch_norm
                    normalizer_params = {'is_training'
                                         : self.is_training
                                         }

                (num_filter, kernel_size, stride) = layer_config
                layer_output = tf.contrib.layers.conv2d(
                    layer_inputs, num_filter, kernel_size, stride,
                    activation_fn=lambda x:tf.nn.leaky_relu(x,alpha=0.2),
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params)

                discrim_layer_list.append(layer_output)
            print('discrim : ', discrim_layer_list[-1].shape)

        layer_inputs = discrim_layer_list[-1]
        layer_inputs = tf.contrib.layers.flatten(layer_inputs)
        discrim_logit = tf.contrib.layers.fully_connected(
            layer_inputs, class_num, None)
        print('discriminator output : ', discrim_logit.shape)

        return discrim_logit



class Disentagled_VAE_CNN_Discrim(Beta_VAE_Discrim):
    def __init__(self, encoder_layer, decoder_layer, input_dim,  z_dim,
                 is_training, batch_size,  beta_s, beta_c,
                 output_prob,
                 learning_rate, optimizer, discriminator_layer, classifier_layer,
                 context_class_num, discrim_lambda, classifier_lambda, deterministic_c, lambda_d_reg):
        super().__init__(encoder_layer, decoder_layer, input_dim, z_dim,
                         is_training, batch_size,  beta_s, beta_c,
                         output_prob,
                         learning_rate, optimizer, discriminator_layer,classifier_layer,
                         context_class_num, discrim_lambda, classifier_lambda,
                         deterministic_c, lambda_d_reg)

    def _build_encoder(self, encoder_layer):
        encoder_layer_list = list()
        encoder_layer_list.append(self.input_data)
        print('encoder input :', self.input_data.shape)

        for layer_num, layer_config in enumerate(encoder_layer):
            (num_filter, kernel_size, stride) = layer_config
            with tf.variable_scope('layer_{}'.format(layer_num)):
                layer_inputs = encoder_layer_list[-1]

                if layer_num == len(encoder_layer) - 1:
                    normalizer_fn = None
                    normalizer_params = None
                else:
                    normalizer_fn = tf.contrib.layers.batch_norm
                    normalizer_params = {'is_training'
                                         : self.is_training
                                         }

                layer_output = tf.contrib.layers.conv2d(
                    layer_inputs, num_filter, kernel_size, stride,
                    activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.2),
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params)

                encoder_layer_list.append(layer_output)

                print('enc : ', encoder_layer_list[-1].shape)

        encoder_output_shape = encoder_layer_list[-1].shape.as_list()[1]

        #last fully connected before latent variable creation

        layer_inputs = encoder_layer_list[-1]
        encoder_output = tf.contrib.layers.flatten(layer_inputs)
        print('encoder output :', encoder_output.shape)
        return encoder_output, encoder_output_shape

    def _build_decoder(self, layer_input, decoder_layer, beta_s, 
                       input_dim, encoder_output_shape):

        # decoder_layer.append(input_dim * input_dim)
        '''
        :param layer_input:
        :param decoder_layer:
        :param beta_s:
        :param final_act_fn:
        :param input_dim:
        :return:
        '''

        ''''
          def _get_deconv2d_output_size(self, input_height, input_width, filter_height,
                                filter_width, row_stride, col_stride, padding_type):
        if padding_type == 'VALID':
          out_height = (input_height - 1) * row_stride + filter_height
          out_width  = (input_width  - 1) * col_stride + filter_width
        elif padding_type == 'SAME':
          out_height = input_height * row_stride
          out_width  = input_width * col_stride
        return out_height, out_width        
        '''


        # layer_input_expand_dim = tf.expand_dims(tf.expand_dims(layer_input, 1), 1)

        # layer_input_expand_dim = tf.expand_dims(layer_input, [1, 2])

        # decoder_layer.append((layer_input.shape[-1], 1, 1))

        decoder_layer_list = list()
        decoder_layer_list.append(layer_input)
        print('decoder input :', layer_input.shape)

        # filter_size = layer_input.shape[-1]

        for layer_num, layer_config in enumerate(decoder_layer):
            (num_filter, kernel_size, stride) = layer_config
            with tf.variable_scope('layer_{}'.format(layer_num)):
                layer_inputs = decoder_layer_list[-1]
                if layer_num == len(decoder_layer) - 1:
                    normalizer_fn = None
                    normalizer_params = None
                    activation_fn = tf.identity
                else:
                    normalizer_fn = tf.contrib.layers.batch_norm
                    normalizer_params = {'is_training' : self.is_training}
                    activation_fn = lambda x: tf.nn.leaky_relu(x, alpha=0.2)

                if layer_num == 0:
                    layer_output = tf.contrib.layers.fully_connected(
                        layer_inputs, num_filter* encoder_output_shape * encoder_output_shape,
                        activation_fn=lambda x:tf.nn.leaky_relu(x, alpha=0.2),
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params)
                    layer_output = tf.reshape(layer_output, (-1, encoder_output_shape,
                                                             encoder_output_shape,
                                                             num_filter))
                    output_size = encoder_output_shape
                else:
                    layer_output = tf.contrib.layers.conv2d_transpose(
                        layer_inputs, num_filter, kernel_size, stride,
                        activation_fn=activation_fn,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params)
                    output_size = output_size * stride

                decoder_layer_list.append(layer_output)

            print('dec : ', decoder_layer_list[-1].shape)

        decoder_output = decoder_layer_list[-1]

        crop = int((output_size - input_dim) / 2)
        if crop > 0:
            decoder_output = decoder_output[:,crop:-crop,crop:-crop,:]

        logit = tf.reshape(decoder_output, (-1, input_dim,
                                                 input_dim, 1))
        print('decoder output :', logit.shape, 'output size', output_size,
              'input dim :', input_dim, 'crop:', crop)

        return logit

    def _build_discriminator(self, layer_input, discriminator_layer, class_num):
        discrim_layer_list = list()
        discrim_layer_list.append(layer_input)

        print('discrimiantor input :', layer_input.shape)
        for layer_num, layer_config in enumerate(discriminator_layer):

            with tf.variable_scope('layer_{}'.format(layer_num)):
                layer_inputs = discrim_layer_list[-1]

                if layer_num == 0:
                    normalizer_fn = None
                    normalizer_params = None
                else:
                    normalizer_fn = tf.contrib.layers.layer_norm
                    normalizer_params = None

                (num_filter, kernel_size, stride) = layer_config
                layer_output = tf.contrib.layers.conv2d(
                    layer_inputs, num_filter, kernel_size, stride,
                    activation_fn=lambda x:tf.nn.leaky_relu(x,alpha=0.2),
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params)

                discrim_layer_list.append(layer_output)
            print('discrim : ', discrim_layer_list[-1].shape)

        layer_inputs = discrim_layer_list[-1]
        layer_inputs = tf.contrib.layers.flatten(layer_inputs)
        print('discrim : ', layer_inputs.shape)
        discrim_logit = tf.contrib.layers.fully_connected(
            layer_inputs, class_num, None)
        print('discriminator output : ', discrim_logit.shape)

        return discrim_logit



