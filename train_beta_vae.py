

import re
import argparse
import os
import time
import numpy as np
import utils
import tensorflow as tf

# from layerwise_discriminator import get_logger, Disentagled_VAE
from model import get_logger, Disentagled_VAE_CNN, Disentagled_VAE_FC
# from tensorflow.python.client import timeline

# tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(0)
'''
error 이상만 찍히도록 : end of sequence 로 생기는 문제 없애기
'''

parser = argparse.ArgumentParser()
parser.add_argument('-m', help='model name', type=str)
parser.add_argument('-b', help = 'batch size', default=256, type=int)
parser.add_argument('-enc', type=int, nargs='+')

parser.add_argument('-input_dim', type=int)
parser.add_argument('-fc_dim', help = 'disentagled partition per layer', type=int)
parser.add_argument('-z_dim', help = 'disentagled partition per layer', type=int)
parser.add_argument('-final_act_fn', type=str)
parser.add_argument('-output_prob', help='probability model of output', type=str)
parser.add_argument('-dec', type=int, nargs='+')

parser.add_argument('-beta', type=float)
# parser.add_argument('-bn', help = 'batch normalization in fully_connected',
#                     action='store_true', default=False)
parser.add_argument('-data_dir', help = 'data directory', type = str)
# parser.add_argument('-test_data_size', help='test_data_size', type=int)
# parser.add_argument('-test_batch_size', help='test_batch_size', type=int)

# parser.add_argument('-disen_act_fn', help='disentagled_activaion_fn', type=str)
# parser.add_argument('-discrim', help='discrimiantor layer structure', type=int, nargs='+')

# parser.add_argument('-num_cycle', help='number of cycles for each epoch', type=int)
parser.add_argument('-option', help='optional specifications', type=str)

parser.add_argument('-gpu_num', help='gpu number to operate', type=int, default=0)

parser.add_argument('-learning_rate', type=float)

parser.add_argument('-optimizer', type=str)

parser.add_argument('-random', help='randomized_latent_classes_of_dsprite',
                    action='store_true',
                    default=False)

parser.add_argument('-restore', help='restore or not', action='store_true',
                    default=False)

class OverfitError(Exception):
    pass

class ModelOverlapError(Exception):
    pass

args = parser.parse_args()

model_name = args.m
batch_size = args.b
input_dim = args.input_dim
encoder_layer = args.enc
z_dim = args.z_dim
fc_dim = args.fc_dim
beta = args.beta
output_prob = args.output_prob
final_act_fn = args.final_act_fn
decoder_layer = args.dec
data_dir = args.data_dir
learning_rate = args.learning_rate
optimizer = args.optimizer
random = args.random
# test_data_size = args.test_data_size
# test_batch_size = args.test_batch_size

option = args.option
gpu_num = args.gpu_num
restore = args.restore

model_config = {'model' : str(model_name),
                'enc' : str(encoder_layer),
                'fc_dim' : fc_dim,
                'z_dim' : z_dim,
                'beta': beta,
                'dec': str(decoder_layer),
                'final_act_fn' : str(final_act_fn),
                'output_prob' : str(output_prob),
                'optim' : optimizer,
                'l_rate' : str(learning_rate),
                'rand' : str(random)
                }

if option:
    model_config['option'] = option



for key, val in model_config.items():
    new_val = re.sub('[\[\]]', '', str(val))
    new_val = re.sub(', ', '-', new_val)
    model_config[key] = new_val


print(model_config)


if model_name == 'FC':
    model = Disentagled_VAE_FC
elif model_name == 'CNN':
    model = Disentagled_VAE_CNN


logger, model_dir = get_logger(data_dir, model_config)


save_dir = data_dir + '/summary/' + model_dir +'/'
if 'checkpoint' in os.listdir(save_dir):
    if restore:
        pass
    else:
        raise ModelOverlapError('Already model exists but no restore flag')

if data_dir == 'cifar-10':
    import cifar_10_inputs
    inputs = cifar_10_inputs.inputs
elif data_dir == 'higgs':
    import higgs_intputs
    inputs = higgs_intputs.inputs
elif data_dir =='mnist':
    import mnist_inputs
    inputs = mnist_inputs.inputs
elif data_dir == 'dsprite':
    import dsprite_inputs
    # inputs = dsprite_inputs.inputs
    print('single_file')
    inputs = dsprite_inputs.inputs_single_file


if final_act_fn == 'relu':
    final_act_fn = tf.nn.relu
elif final_act_fn == 'tanh':
    final_act_fn = tf.nn.tanh
elif final_act_fn == 'sigmoid':
    final_act_fn = tf.nn.sigmoid
elif final_act_fn == 'linear':
    final_act_fn = None


if model_name == 'CNN':
    tmp = [encoder_layer[i:i+3] for i in range(0, len(encoder_layer), 3)]
    encoder_layer = tmp

    tmp = [decoder_layer[i:i+3] for i in range(0, len(decoder_layer), 3)]
    decoder_layer = tmp


train_inputs, train_init_op = inputs(batch_size, 5, random)
bn_phase = tf.placeholder(tf.bool)
with tf.variable_scope('Model'):
    trn_model = model(
        encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
        bn_phase, batch_size, train_inputs, final_act_fn, beta, output_prob,
        learning_rate, optimizer)


# test_inputs, test_init_op = inputs(test_batch_size, 5)
# with tf.variable_scope('Model'):
#     test_model = model(
#         encoder_layer, decoder_layer, input_dim, fc_dim, z_dim,
#         False, batch_size, train_inputs, target_fn, beta, output_prob)

# with tf.name_scope('Test'):
#     test_inputs, test_init_op = inputs('test', test_batch_size, num_epoch, 5)
#     with tf.variable_scope('Model', reuse=True):
#         test_model = model(encoder_layer, decoder_layer, input_dim, fc_dim,
#                                      z_dim,
#             False, batch_size, test_inputs, target_fn, beta, output_prob)



config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True


with tf.Session(config=config) as sess:
    tf.set_random_seed(1)

    saver = tf.train.Saver(max_to_keep=50)
    saver_periodical = tf.train.Saver(max_to_keep=50)

    test_writer = tf.summary.FileWriter(logdir=save_dir, graph=sess.graph)
    if restore:
        path = tf.train.latest_checkpoint(save_dir)
        epoch = int(re.search('model.ckpt-(\d+)', path).group(1))
        saver.restore(sess, path)
        print('previous epoch', epoch)
    else:
        epoch = 0
        sess.run(tf.global_variables_initializer())

    # config = projector.ProjectorConfig()
    # embedding = config.embeddings.asdd()
    # embedding.tensor_name = trn_model.embedding_var.name
    # embedding.metadata_path = os.path.join(data_dir, 'metadata.tsv')



    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    best_loss = 1e4
    test_loss = np.array([1e4, 1e4, 1e4])
    max_patience = 30
    max_epoch = 200
    patience = 0

    # tf.get_default_graph().finalize()
    ''''
    finalize graph
    '''

    # from tensorflow.contrib.tensorboard.plugins import projector
    # config = projector.ProjectorConfig()
    # print('hello')
    # c = sess.run(trn_model.decoder_last)
    # print(c.shape)
    # a = sess.run(trn_model.recon)
    # print(a.shape)
    # b= sess.run(trn_model.input_data)
    # print(b.shape)



    # loss_update_op_trn = tf.get_collection('loss_update', 'Train')
    # losS_update_op_test = tf.get_collection('loss_update', 'Test')
    # stream_loss_test = [tf.get_collection('streaming_loss', 'Train'), tf.get_collection(
    #     'streaming_loss', 'Test')]


    trn_eval = utils.Eval(trn_model, train_init_op, sess, logger,test_writer,
                          saver, saver_periodical, save_dir, max_patience, max_epoch)

    try:
        while True:
            prev = time.time()

            # sess.run(tf.local_variables_initializer())

            trn_eval.train()

            epoch = epoch + 1

            # while True:
                # try:
                #     # _, _, global_step_val = sess.run([trn_model.train_op, trn_eval.loss_update_op,
                #     #                                   trn_model.global_step])
                #     #
                #     # if global_step_val % 1000 == 0:
                #     #     trn_loss = sess.run(trn_eval.stream_loss)
                #     #     summary_val = sess.run(trn_model.loss_summaries, feed_dict={
                #     #         trn_model.loss_values: trn_loss})
                #     #     test_writer.add_summary(summary_val, epoch)
                #
                # except tf.errors.OutOfRangeError:
                #     break

            # trn_loss = sess.run(stream_loss[0])
            # summary_val = sess.run(trn_model.loss_summaries, feed_dict={
            #     trn_model.loss_values:trn_loss})

            now = time.time()
            minutes = (now - prev) / 60
            message = 'epoch took {} min'.format(minutes)

            logger.info(message + '\n')
            trn_eval.eval_score_and_save_model(epoch)
            print(model_config)

            # test_eval.eval_score.and_save_model(epoch)

            # test_eval.eval_score_and_save_model(epoch)
            # sess.run(test_init_op)
            #
            # image_flag=True
            # while True:
            #     try:
            #         if image_flag:
            #             fetch = [loss_update_op[1], test_model.image_summaries]
            #             _, img_summary_val = sess.run(fetch)
            #             test_writer.add_summary(img_summary_val, global_step=epoch)
            #             image_flag = False
            #         else:
            #             fetch = [loss_update_op[1]]
            #             sess.run(fetch)
            #     except tf.errors.OutOfRangeError:
            #         break
            # test_loss = sess.run(stream_loss[1])
            # test_summary_val = sess.run(test_model.loss_summaries, feed_dict={
            #     test_model.loss_values: test_loss})
            # test_writer.add_summary(test_summary_val, global_step=epoch)
            #
            # result = 'epoch {} | trn_loss : {} test loss : {} '.format(epoch, trn_loss, test_loss)
            #
            # logger.info(message + '\n')
            # logger.info(result + '\n')


            # if global_step_val > 5000 and test_loss[1] > trn_loss_batch[1] * 1.5:
            #     logger.info('Overfit')
            #     raise OverfitError('Overfit')

            # if test_loss[1] < best_loss:
            #     print('better model in recon loss')
            #     saver.save(sess, '{}/summary/{}/model.ckpt'.format(data_dir,
            #                                                            model_dir),
            #                      global_step=epoch)
            #     best_loss = test_loss[0]
            #     patience = 0
            # else:
            #     patience += 1
            #     if patience == max_patience:
            #         logger.info('Max patience reached')
            #         raise MaxPatienceError('Max patience')
    except utils.MaxPatienceError:
        logger.info('Done training -- max patience reached')
    sess.close()

