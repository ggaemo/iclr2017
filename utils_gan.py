

import tensorflow as tf
import itertools
import collections
import numpy as np

'''


s permutation: all random
c permutation: within class



'''
class MaxPatienceError(Exception):
    pass

class MaxEpochError(Exception):
    pass

class Eval():

    def __init__(self, model, inputs, init_op, sess, logger,
                 writer, saver, saver_periodical, save_dir, max_patience, max_epoch,
                 perm_change, context_class_num):
        self.model = model
        self.inputs = inputs
        self.init_op = init_op
        self.sess = sess
        self.logger = logger
        self.test_writer = writer
        self.saver = saver
        self.saver_periodical = saver_periodical
        self.patience = 0
        self.max_patience = max_patience
        self.max_epoch = max_epoch
        self.best_loss = 1e4
        self.save_dir = save_dir
        self.perm_change = perm_change
        self.context_class_num = context_class_num
        self.loss_update_op = tf.get_collection('loss_update')
        self.stream_loss = tf.get_collection('streaming_loss')

    def train(self):

        self.sess.run(self.init_op)

        count = 0
        while True:
            try:
                 _, a, l = self.sess.run([self.model.train_op, self.model.recon,
                                       self.model.recon_loss],
                                         feed_dict={
                     self.model.is_training : True})
                 count += 1
                 if count % 1000 == 0:
                    self.logger.info('max val, minval, recon_loss : {} {} {}'.format(
                                     np.max(a),np.min(a), l))

            except tf.errors.OutOfRangeError:
                print('train out of range')
                break

        print('done training')

    def eval_score_and_save_model(self, epoch):

        '''

        To add different types of images to tensorboard images are added to the
        summary by a chance of 0.01%

        :param epoch:
        :return:

        '''
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.init_op)

        fetch = [self.loss_update_op, self.model.image_summaries]
        _, img_summary_val = self.sess.run(fetch, feed_dict={
            self.model.is_training: False})
        self.test_writer.add_summary(img_summary_val, global_step=epoch)

        while True:

            try:
                fetch = [self.loss_update_op]
                self.sess.run(fetch, feed_dict={self.model.is_training:False})

            except tf.errors.OutOfRangeError:
                print('eval out of range')
                break
        test_loss = self.sess.run(self.stream_loss)
        test_summary_val = self.sess.run(self.model.loss_summaries, feed_dict={
            self.model.loss_values: test_loss})
        self.test_writer.add_summary(test_summary_val, global_step=epoch)
        self.test_writer.flush()

        result = 'epoch {} | loss : vae, recon, c, s, discrim {} '.format(epoch, test_loss)

        self.logger.info(result + '\n')

        self.save_model(test_loss, epoch)

    def save_model(self, test_loss, epoch):

        if epoch > self.max_epoch:
            self.logger.info('Max Epoch reached')
            raise MaxEpochError('Max Epoch')
        if test_loss[1] < self.best_loss:  # second term refers to recon_loss
            print('better model in recon loss')
            self.saver.save(self.sess, self.save_dir+'model.ckpt',
                       global_step=epoch)
            self.best_loss = test_loss[1]
            self.patience = 0
        elif epoch % 30 == 0:
            print('save every 30 epochs')
            self.saver_periodical.save(self.sess, self.save_dir + 'model_periodical.ckpt',
                            global_step=epoch)
        else:
            self.patience += 1
            if self.patience == self.max_patience:
                self.logger.info('Max patience reached')
                raise MaxPatienceError('Max patience')



class Eval_discrim(Eval):
    def __init__(self, model, inputs, init_op, sess, logger,
                     writer, saver, saver_periodical, save_dir, max_patience, max_epoch,
                 perm_change, context_class_num):
        super().__init__(model, inputs, init_op, sess, logger,
                     writer, saver, saver_periodical, save_dir, max_patience, max_epoch,
                         perm_change, context_class_num)

        candidate_perm = list()
        for class_perm in itertools.permutations(range(self.context_class_num)):
            flag = True
            for i in range(self.context_class_num):
                if class_perm[i] == i:
                    flag = False  # to avoid the permutation where class arrangement matches
                    # the
                    # original class
                    break
            if flag:
                candidate_perm.append(class_perm)

        self.class_shuffle = itertools.cycle(candidate_perm)

    def _get_permutation(self):
        perm_dict = collections.OrderedDict()
        for i in range(self.context_class_num):
            perm_dict[i] = np.repeat(np.random.permutation([i * 2, i * 2 + 1]),
                                     int(
                                         self.model.batch_size / self.context_class_num / 2))

        permutation_c = np.concatenate([x for x in perm_dict.values()])

        print('permutation_c:', permutation_c)

        perm_list = list()

        class_shuffle_val = next(self.class_shuffle)

        for class_idx in class_shuffle_val:
            perm_list.append(perm_dict[class_idx])

        permutation_s = np.concatenate(perm_list)
        print('permutation class shuffle:', class_shuffle_val)
        print('permutation_s:', permutation_s)

        return permutation_c, permutation_s

    def train(self):

        self.sess.run(self.init_op)

        def _report(config):
            _, recon_loss, dis_gen_loss, dis_ad_loss, dis_recon_acc, \
            dis_input_acc, class_loss = \
                self.sess.run([
                    self.model.train_op_adv,
                    self.model.recon_loss,
                    self.model.discrim_generative_loss,
                    self.model.discrim_adversarial_loss_g,
                    self.model.discrim_recon_s_acc,
                    self.model.discrim_input_acc,
                    self.model.classifier_loss],
                    feed_dict={
                        self.model.is_training: True,
                        self.model.permutation_s: permutation_s,
                        self.model.permutation_c: permutation_c
                    })
            print(
                '{} epoch {} sub_count {} : recon {:.2f} d_g {:.2f} '
                'd_a {:.2f} c {:.2f} d_input_acc {:.2f} d_recon_acc {:.2f}'.format(config,
                                                                              count,
                                                                   sub_count, recon_loss,
                                                                   dis_gen_loss,
                                                                   dis_ad_loss,
                                                                   class_loss,
                                                                   dis_input_acc,
                                                                   dis_recon_acc))



        count = 0

        dis_gen_loss = 1e3
        dis_ad_loss = 1e3
        recon_loss = 1e4
        class_loss = 1e4

        while True:

            if count % self.perm_change == 0:

                # perm0 = np.repeat(np.random.permutation([0, 1]), int(self.model.batch_size /
                #                                                      3 /2))
                # perm1 = np.repeat(np.random.permutation([2, 3]), int(self.model.batch_size /
                #                                                      3 / 2))
                # perm2 = np.repeat(np.random.permutation([4, 5]), int(self.model.batch_size /
                #                                                      3 / 2))
                #
                # perm_dict = {0 : perm0,
                #              1 : perm1,
                #              2 : perm2}
                #
                # permutation_c = np.concatenate([perm0, perm1, perm2])
                #
                # print('permutation_c:', permutation_c)
                #
                # perm_list = list()
                #
                # class_shuffle_val = next(self.class_shuffle)
                #
                # for class_idx in class_shuffle_val:
                #     perm_list.append(perm_dict[class_idx])
                #
                # permutation_s = np.concatenate(perm_list)
                # print('permutation class shuffle:', class_shuffle_val)
                # print('permutation_s:', permutation_s)
                permutation_c, permutation_s = self._get_permutation()


            run_period = 1
            k = 1


            try:
                inputs = self.sess.run(self.inputs)
                input_data = inputs['img']
                latent = inputs['latent']
                sub_count = 0

                # vae RECON
                sub_count = 0
                while (True):
                    sub_count += 1

                    if sub_count == 2:
                        break
                    # if sub_count % 50 == 0:
                    #     permutation_s = _change_permutation_s(sub_count)

                    for _ in range(run_period):
                        _, recon_loss, dis_gen_loss, dis_ad_loss, class_loss, \
                        class_loss_gen\
                            = \
                            self.sess.run([
                                self.model.train_op_vae,
                                self.model.recon_loss,
                                self.model.discrim_generative_loss,
                                self.model.discrim_adversarial_loss_g,
                                self.model.classifier_loss,
                                self.model.classifier_loss_gen],
                                feed_dict={
                                    self.model.is_training: True,
                                    self.model.permutation_s: permutation_s,
                                    self.model.permutation_c: permutation_c,
                                    self.model.input_data: input_data,
                                    self.model.latent: latent
                                })



                #classifier train
                sub_count = 0
                while (True):
                    sub_count += 1
                    # if sub_count % 50 == 0:
                    #     permutation_s = _change_permutation_s(sub_count)

                    if sub_count ==2 :
                        break
                    for _ in range(run_period):
                        _, class_loss = self.sess.run([self.model.train_op_classifier,
                                       self.model.classifier_loss],
                                      feed_dict={
                                          self.model.is_training: True,
                                          self.model.permutation_s: permutation_s,
                                          self.model.permutation_c: permutation_c,
                            self.model.input_data:input_data,
                            self.model.latent:latent
                                      })
                # _report('classifier_train')




                # vae classifier
                        # vae
                sub_count = 0
                while (True):
                    sub_count += 1

                    if sub_count == 2:
                        break
                    # if sub_count % 50 == 0:
                    #     permutation_s = _change_permutation_s(sub_count)
                    #
                    for _ in range(run_period):
                        _, recon_loss, dis_gen_loss, dis_ad_loss, class_loss, \
                        class_loss_gen\
                            = \
                            self.sess.run([
                                self.model.train_op_classifier_gen,
                                self.model.recon_loss,
                                self.model.discrim_generative_loss,
                                self.model.discrim_adversarial_loss_g,
                                self.model.classifier_loss,
                                self.model.classifier_loss_gen],
                                feed_dict={
                                    self.model.is_training: True,
                                    self.model.permutation_s: permutation_s,
                                    self.model.permutation_c: permutation_c,
                                    self.model.input_data: input_data,
                                    self.model.latent: latent
                                })

                # vae_discrim
                sub_count = 0
                while (True):
                    sub_count += 1

                    if sub_count == 2:
                        break
                    # if sub_count % 50 == 0:
                    #     permutation_s = _change_permutation_s(sub_count)

                    if dis_ad_loss < dis_gen_loss:
                        _, dis_gen_loss, dis_ad_loss = \
                            self.sess.run([
                                self.model.train_op_gen,
                                self.model.discrim_generative_loss,
                                self.model.discrim_adversarial_loss_g],
                                feed_dict={
                                    self.model.is_training: True,
                                    self.model.permutation_s: permutation_s,
                                    self.model.permutation_c: permutation_c,
                            self.model.input_data:input_data,
                            self.model.latent:latent
                                })
                    else:
                        _, dis_gen_loss, dis_ad_loss = self.sess.run(
                            [self.model.train_op_adv_g,
                             self.model.discrim_generative_loss,
                             self.model.discrim_adversarial_loss_g
                             ],
                            feed_dict={
                                self.model.is_training: True,
                                self.model.permutation_s: permutation_s,
                                self.model.permutation_c: permutation_c,
                                self.model.input_data: input_data,
                                self.model.latent: latent

                            })


                        _, dis_gen_loss, dis_ad_loss = self.sess.run(
                            [self.model.train_op_adv_r,
                             self.model.discrim_generative_loss,
                             self.model.discrim_adversarial_loss_g
                             ],
                            feed_dict={
                                self.model.is_training: True,
                                self.model.permutation_s: permutation_s,
                                self.model.permutation_c: permutation_c,
                                self.model.input_data: input_data,
                                self.model.latent: latent
                            })

                    # if sub_count % 100 == 0:
                # _report('after_vae_recon_gen_class')



                        # _report('after_vae_recon_gen_class')

                self.logger.info('whole cycle : recon {:.2f} d_g {:.2f} '
                                 'd_a {:.2f} c {:.2f} c_g {:.2f}'.format(recon_loss,
                                                         dis_gen_loss,
                                                         dis_ad_loss,
                                                         class_loss, class_loss_gen))

                count += 1
                # if input('hello') == 'yes':
                #     break

            except tf.errors.OutOfRangeError:
                self.logger.info('train out of range')
                break


        print('done training')

    def eval_score_and_save_model(self, epoch):

        '''

        To add different types of images to tensorboard images are added to the
        summary by a chance of 0.01%

        :param epoch:
        :return:

        '''
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.init_op)


        permutation_c, permutation_s = self._get_permutation()

        while True:

            # eval just range
            coin = np.random.binomial(1, 0.01)
            try:
                inputs = self.sess.run(self.inputs)
                input_data = inputs['img']
                latent = inputs['latent']

                if coin == 1:
                    fetch = [self.loss_update_op, self.model.image_summaries]
                    _, img_summary_val = self.sess.run(fetch, feed_dict={
                        self.model.is_training: False,
                        self.model.permutation_s: permutation_s,
                        self.model.permutation_c: permutation_c,
                        self.model.input_data: input_data,
                        self.model.latent: latent
                        })

                    self.test_writer.add_summary(img_summary_val, global_step=epoch)
                else:
                    fetch = [self.loss_update_op]
                    self.sess.run(fetch, feed_dict={self.model.is_training:False,
                                                     self.model.permutation_s:permutation_s,
                                                    self.model.permutation_c:permutation_c,
                                                    self.model.input_data: input_data,
                                                    self.model.latent: latent
                                                    })
            except tf.errors.OutOfRangeError:
                print('eval out of range')
                break
        test_loss = self.sess.run(self.stream_loss)
        test_summary_val = self.sess.run(self.model.loss_summaries, feed_dict={
            self.model.loss_values: test_loss})
        self.test_writer.add_summary(test_summary_val, global_step=epoch)
        self.test_writer.flush()

        result = 'epoch {} | loss : recon {}, vae {}, latent_s {}, d_gen {}, d_adv {}, ' \
                 'd_recon_s_acc {}, d_recon_c_acc {}, d_input {}, c_input {}, c_recon_s ' \
                 '{}, c_input_acc ' \
                 '{} c_recon_s_acc {} c_recon_c_acc {}'.format(
            epoch, *test_loss)

        self.logger.info(result + '\n')

        self.save_model(test_loss, epoch)