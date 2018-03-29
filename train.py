

import argparse
import os
import re
import time

import numpy as np
import tensorflow as tf

import utils_train_eval as utils
from misc import get_logger
from model import Disentagled_VAE_FC_Discrim, Disentagled_VAE_CNN_Discrim

# tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(0)
'''
error 이상만 찍히도록 : end of sequence 로 생기는 문제 없애기
'''

parser = argparse.ArgumentParser()
parser.add_argument('-m', help='model name', type=str)
parser.add_argument('-b', help = 'batch size has to be divisor of total batch for the '
                                 'permutation to work',
                    default= 3 * 2 * 20,
                    type=int)

# mnist : 10 * 10
# dsprites : 3 * 2 * 20
parser.add_argument('-enc', type=int, nargs='+')

parser.add_argument('-input_dim', type=int)
parser.add_argument('-z_dim', help = 'disentagled partition per layer', type=int)
parser.add_argument('-output_prob', help='probability model of output', type=str)
parser.add_argument('-dec', type=int, nargs='+')

parser.add_argument('-beta_s', type=float)
parser.add_argument('-beta_c', type=float)

parser.add_argument('-data_dir', help = 'data directory', type = str)

parser.add_argument('-option', help='optional specifications', type=str)

parser.add_argument('-learning_rate', type=float)

parser.add_argument('-optimizer', type=str)

parser.add_argument('-discriminator_layer', type=int, nargs='+')
parser.add_argument('-classifier_layer', type=int, nargs='+')

parser.add_argument('-context_class_num', type=int)
parser.add_argument('-d_lambda', help='discriminator loss weight', type=float)
parser.add_argument('-c_lambda', help='classifier loss weight', type=float)

parser.add_argument('-lambda_d_reg', help='lambda for critic regualrization',
                    type=float, default=10)
parser.add_argument('-deterministic_c', help='deterministic c', action='store_true',
                    default=False)

parser.add_argument('-perm_change', help='number of batches before changing permutation',
                    type=float)

parser.add_argument('-restore', help='restore or not', action='store_true',
                    default=False)

parser.add_argument('--debug', help='debug mode or not', action='store_true',
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
beta_s = args.beta_s
beta_c = args.beta_c
output_prob = args.output_prob
decoder_layer = args.dec
data_dir = args.data_dir
learning_rate = args.learning_rate
optimizer = args.optimizer
discriminator_layer = args.discriminator_layer
classifier_layer = args.classifier_layer
context_class_num = args.context_class_num
discrim_lambda = args.d_lambda
class_lambda = args.c_lambda
deterministic_c = args.deterministic_c
perm_change = args.perm_change
lambda_d_reg = args.lambda_d_reg

option = args.option

restore = args.restore

debug = args.debug



model_config = {'model' : str(model_name),
                'enc' : str(encoder_layer),
                'dec' : str(decoder_layer),
                'z_dim' : z_dim,
                'beta': str(beta_s)+'_'+str(beta_c),
                'discrim' : str(discriminator_layer),
                'classifier' : str(classifier_layer),
                'optim': str(optimizer),
                'perm' : str(perm_change)
                }

if option:
    model_config['option'] = option



for key, val in model_config.items():
    new_val = re.sub('[\[\]]', '', str(val))
    new_val = re.sub(', ', '-', new_val)
    model_config[key] = new_val


print(model_config)


if model_name == 'FC':
    model = Disentagled_VAE_FC_Discrim
elif model_name == 'CNN':
    model = Disentagled_VAE_CNN_Discrim


logger, model_dir = get_logger(data_dir, model_config)


save_dir = data_dir + '/summary/' + model_dir +'/'
if 'checkpoint' in os.listdir(save_dir):
    if restore:
        pass
    else:
        raise ModelOverlapError('Already model exists but no restore flag')

if data_dir == 'higgs':
    import higgs_intputs
    inputs = higgs_intputs.inputs
elif data_dir =='mnist':
    from mnist import mnist_inputs
    inputs = mnist_inputs.inputs_single_file
elif data_dir == 'dsprites':
    from dsprites import dsprites_inputs
    print('single_file')
    inputs = dsprites_inputs.inputs_single_file


if model_name == 'CNN':
    tmp = [encoder_layer[i:i+3] for i in range(0, len(encoder_layer), 3)]
    encoder_layer = tmp

    tmp = [decoder_layer[i:i+3] for i in range(0, len(decoder_layer), 3)]
    decoder_layer = tmp

tmp = [discriminator_layer[i:i+3] for i in range(0, len(discriminator_layer), 3)]
discriminator_layer = tmp

tmp = [classifier_layer[i:i+3] for i in range(0, len(classifier_layer), 3)]
classifier_layer = tmp

train_inputs, train_init_op = inputs(batch_size, 10)
bn_phase = tf.placeholder(tf.bool)

with tf.variable_scope('Model'):
    trn_model = model(
        encoder_layer, decoder_layer, input_dim, z_dim,
        bn_phase, batch_size, beta_s, beta_c, output_prob,
        learning_rate, optimizer, discriminator_layer, classifier_layer,
        context_class_num, discrim_lambda, class_lambda, deterministic_c, lambda_d_reg)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True


from tensorflow.python import debug as tf_debug


with tf.Session(config=config) as sess:
    if debug:
        debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    else:
        debug_sess = sess

    saver = tf.train.Saver(max_to_keep=5)
    saver_periodical = tf.train.Saver(max_to_keep=5)

    test_writer = tf.summary.FileWriter(logdir=save_dir, graph=sess.graph)
    if restore:
        path = tf.train.latest_checkpoint(save_dir)
        epoch = int(re.search('model.ckpt-(\d+)', path).group(1))
        saver.restore(sess, path)
        print('previous epoch', epoch)
    else:
        epoch = 0
        sess.run(tf.global_variables_initializer())


    best_loss = 1e4
    test_loss = np.array([1e4, 1e4, 1e4])
    max_patience = 2000
    max_epoch = 1000
    patience = 0

    trn_eval = utils.Eval_discrim(trn_model, train_inputs, train_init_op, sess, logger,
                                  test_writer,
                                  saver, saver_periodical, save_dir, max_patience,
                                  max_epoch, perm_change, context_class_num, debug_sess)

    try:
        while True:
            prev = time.time()

            trn_eval.train()

            epoch = epoch + 1


            now = time.time()
            minutes = (now - prev) / 60
            message = 'epoch took {} min'.format(minutes)

            logger.info(message + '\n')
            trn_eval.eval_score_and_save_model(epoch)
            print(model_config)

    except utils.MaxPatienceError:
        logger.info('Done training -- max patience reached')
    sess.close()



