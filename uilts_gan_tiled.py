import tensorflow as tf

import pandas as pd
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

    def __init__(self, model, init_op, sess, logger,
                 writer, saver, saver_periodical, save_dir, max_patience, max_epoch):
        self.model = model
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
            self.best_loss = test_loss[0]
            self.patience = 0
        elif epoch % 20 == 0:
            print('save every 30 epochs')
            self.saver_periodical.save(self.sess, self.save_dir + 'model_periodical.ckpt',
                            global_step=epoch)
        else:
            self.patience += 1
            if self.patience == self.max_patience:
                self.logger.info('Max patience reached')
                raise MaxPatienceError('Max patience')



class Eval_discrim(Eval):
    def __init__(self, model, init_op, sess, logger,
                     writer, saver, saver_periodical, save_dir, max_patience, max_epoch):
        super().__init__(model, init_op, sess, logger,
                     writer, saver, saver_periodical, save_dir, max_patience, max_epoch)
    def train(self):

        self.sess.run(self.init_op)





        count = 0
        while True:

            perm0 = np.random.permutation([0, 1] * int(
                self.model.batch_size / 3 / 2))
            perm1 = np.random.permutation([2, 3] * int(
                self.model.batch_size / 3 / 2))
            perm2 = np.random.permutation([4, 5] * int(
                self.model.batch_size / 3 / 2))
            permutation_c = np.concatenate([perm0, perm1, perm2])

            permutation_s = np.random.permutation(range(int(self.model.batch_size)))

            # perm0 = np.random.permutation(range(int(self.model.batch_size / 3)))
            # perm1 = np.random.permutation(
            #     range(int(self.model.batch_size / 3))) + int(self.model.batch_size/3)
            # perm2 = np.random.permutation(range(int(self.model.batch_size / 3))) + \
            #         int(self.model.batch_size/3) * 2
            # permutation_c = np.concatenate([perm0, perm1, perm2])
            try:

                # self.partitioned_c = tf.dynamic_partition(self.z_v_c,
                #                                           self.permutation_original,
                #                                           class_num)
                # self.permutation_c_partitioned = tf.dynamic_partition(self.permutation_c,
                #                                                       self.permutation_original,
                #                                                       class_num)
                # self.shuffled_content = tf.dynamic_stitch(self.permutation_c_partitioned,
                #                                           self.partitioned_c)

                # a, b, c, d, e = self.sess.run([self.model.partitioned_c,
                #                       self.model.permutation_c_partitioned,
                #                       self.model.shuffled_content,self.model.z_v_c,
                #                             self.model.z_v_s],
                #                      feed_dict ={self.model.permutation_s: permutation_s,
                #                          self.model.permutation_c: permutation_c,
                #                          self.model.permutation_original:
                #                              permutation_original})

                _, l, d_l, pred = self.sess.run([
                    self.model.train_op,self.model.recon_loss,
                    self.model.discrim_loss,self.model.predict],
                                     feed_dict={
                                         self.model.is_training : True,
                                         self.model.permutation_s: permutation_s,
                                         self.model.permutation_c: permutation_c,
                                     })
                count += 1
                if count % 1000 == 0:
                    self.logger.info('recon_loss : {} d_l {} '.format(l,d_l))
                    # print('pred\n', pd.Series(pred).value_counts(
                    #     sort=False))

            except tf.errors.OutOfRangeError:
                self.logger.info('train out of range')
                break

        print('done training')

    def train_separate(self, interval):

        self.sess.run(self.init_op)

        count = 0
        while True:
            perm0 = np.random.permutation(range(int(self.model.batch_size / 3)))
            perm1 = np.random.permutation(
                range(int(self.model.batch_size / 3))) + self.model.batch_size
            perm2 = np.random.permutation(range(int(self.model.batch_size / 3))) + \
                    self.model.batch_size * 2
            permutation_s = np.concatenate([perm0, perm1, perm2])
            perm0 = np.random.permutation(range(int(self.model.batch_size / 3)))
            perm1 = np.random.permutation(
                range(int(self.model.batch_size / 3))) + self.model.batch_size
            perm2 = np.random.permutation(range(int(self.model.batch_size / 3))) + \
                    self.model.batch_size * 2
            permutation_c = np.concatenate([perm0, perm1, perm2])


            try:
                 _, l = self.sess.run([self.model.train_op_vae,
                                       self.model.recon_loss],
                                         feed_dict={
                                             self.model.is_training : True,
                                             self.model.permutation_s: permutation_s,
                                             self.model.permutation_c: permutation_c})

                 if count % interval == 0:
                     for y in range(interval):
                        _, d_l, pred = self.sess.run([
                            self.model.train_op_discrim,
                                             self.model.discrim_loss,
                                                             self.model.predict],
                                                            feed_dict={
                        self.model.is_training: False,
                        self.model.permutation: permutation})

                        if y % int(int(interval) / 3) == 0:
                            self.logger.info('discrim_loss : {}'.format(d_l))
                            # print('pred\n', pd.Series(pred).value_counts(
                            #     sort=False))

                 count += 1
                 # if count % 1000 == 0:
                 #    self.logger.info('recon_loss : {}'.format(l))

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

        permutation_original = np.repeat([0, 1, 2], int(self.model.batch_size / 3))

        while True:
            permutation = np.arange(self.model.batch_size, dtype=np.int32)
            # eval just range
            coin = np.random.binomial(1, 0.01)
            try:
                if coin == 1:
                    fetch = [self.loss_update_op, self.model.image_summaries]
                    _, img_summary_val = self.sess.run(fetch, feed_dict={
                        self.model.is_training: False,
                        self.model.permutation_s: permutation,
                        self.model.permutation_c: permutation,
                        self.model.permutation_original:permutation_original})

                    self.test_writer.add_summary(img_summary_val, global_step=epoch)

                fetch = [self.loss_update_op]
                self.sess.run(fetch, feed_dict={self.model.is_training:False,
                                                self.model.permutation_s:permutation,
                                                self.model.permutation_c:permutation,
                                                self.model.permutation_original:permutation_original})

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