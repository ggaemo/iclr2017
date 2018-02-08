import argparse
import os
import numpy as np
import tensorflow as tf
import time

from network_regularization_2 import get_logger, RegPartModel, RegPartWeightModel
import regularization_losses


# from tensorflow.python.client import timeline



parser = argparse.ArgumentParser()
parser.add_argument('-b', help = 'batch size', default=256, type=int)
parser.add_argument('-n', help = 'number of epochs', default=10000, type=int)
parser.add_argument('-nl', help = 'layer dimensions of normal fully connected layers',
                    type=int, nargs= '+')
parser.add_argument('-reg', help = 'regularization type', type=str)
parser.add_argument('-np', help = 'reg NN output partition per layer', type=int, nargs='+')
parser.add_argument('-no', help = 'reg NN output dimension per layer', type=int, nargs='+')
parser.add_argument('-reg_w', help = 'regularzation loss weight(lambda)', type=float)
parser.add_argument('-bn', help = 'batch normalization in fully_connected', action='store_true')
parser.add_argument('-data_dir', help = 'data directory', type = str)
parser.add_argument('-test_data_size', help='test_data_size', type=int)
parser.add_argument('-test_batch_size', help='test_batch_size', type=int)
parser.add_argument('-huber_target', help='huber loss target', type=float, default = 1)
parser.add_argument('-huber_delta', help='huber loss delta', type=float, default = 1)
parser.add_argument('-num_target_class', help='number of classes in target', type=int,
                    default = 10)
parser.add_argument('-num_cycle', help='number of cycles for each epoch', type=int)
parser.add_argument('-option', help='optional specifications', type=str)

parser.add_argument('-gpu_num', help='gpu number to operate', type=int, default=0)

class MaxPatienceError(Exception):
    pass

args = parser.parse_args()

batch_size = args.b
num_epoch = args.n
num_normal_layer = args.nl
num_partition_by_layer = args.np
reg_num_output_by_layer = args.no
is_bn = args.bn
reg_type = args.reg
reg_weight = args.reg_w
data_dir = args.data_dir
test_data_size = args.test_data_size
test_batch_size = args.test_batch_size
huber_target = args.huber_target
huber_delta = args.huber_delta
num_target_class = args.num_target_class
option = args.option
num_cycle = args.num_cycle
gpu_num = args.gpu_num

model_config = {'regType' : reg_type,
                'normal_layer' : str(num_normal_layer),
                'reg_layer' : '{0}_{1}'.format(num_partition_by_layer,
                                              reg_num_output_by_layer),
                'fc_bn' : is_bn,
                'reg_weight' : reg_weight,
                'delta' : huber_delta}

if option:
    model_config['option'] = option

logger, model_dir = get_logger(data_dir, model_config)

if not os.path.exists(data_dir+'/model/'+model_dir):
    os.makedirs(data_dir+'/model/'+model_dir)

if data_dir == 'cifar-10':
    import cifar_10_inputs

    inputs = cifar_10_inputs.inputs
elif data_dir == 'higgs':
    import higgs_intputs
    inputs = higgs_intputs.inputs

if reg_type == 'DotHuber':
    reg_loss = regularization_losses.DotHuber(reg_weight, huber_target, huber_delta)
elif reg_type == 'Dot':
    reg_loss = regularization_losses.Dot(reg_weight)
elif reg_type == 'Orthogonal':
    reg_loss = regularization_losses.Orthogonal(reg_weight)
elif reg_type == 'Hinge':
    reg_loss = regularization_losses.Hinge(reg_weight)
else:
    reg_loss = None

print(reg_loss)

with tf.device('/gpu:{}'.format(gpu_num)):
    with tf.name_scope('Train'):
        train_inputs, train_targets = inputs('train', batch_size, num_epoch, 1)
        with tf.variable_scope('Model'):
            trn_model = RegPartWeightModel(reg_loss, num_normal_layer,
                                     reg_num_output_by_layer, num_target_class,
                                     True, is_bn, train_inputs, train_targets)

    with tf.name_scope('Test'):
        test_inputs, test_targets = inputs('test', batch_size, num_epoch, 1)
        with tf.variable_scope('Model', reuse=True):
            test_model = RegPartWeightModel(reg_loss, num_normal_layer,
                                     reg_num_output_by_layer, num_target_class,
                                     False, is_bn, test_inputs, test_targets)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

with tf.Session(config=config) as sess:
    tf.set_random_seed(1)
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(logdir='{}/summary/{}/test'.format(data_dir,
        model_dir), graph=sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    best_loss = 1e4
    max_patience = 10
    patience = 0

    tf.get_default_graph().finalize()
    ''''
    finalize graph
    '''


    try:
        prev = time.time()
        while not coord.should_stop():

            _, global_step_val = sess.run([trn_model.train_op, trn_model.global_step])

            # _, global_step_val = sess.run([trn_model.train_op, trn_model.global_step],
            #                               options=options,
            #                               run_metadata=run_metadata
            #                               )

            # if global_step_val % 100 == 0:
            #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #     with open('timeline_step_%d.json' % global_step_val, 'w') as f:
            #         f.write(chrome_trace)

            if global_step_val % 500 == 0:
                summary_val = sess.run(merged)
                test_writer.add_summary(summary_val, global_step=global_step_val)

            if global_step_val % 1000 == 0:

                trn_acc, trn_loss = sess.run([trn_model.accuracy, trn_model.loss])

                now = time.time()
                minutes = (now - prev) / 60
                message = 'took {} min, running at {} samples / min'.format(minutes,
                                                                            1000 * batch_size /
                                                                            minutes)


                test_loss = 0
                test_acc = 0
                for _ in range(test_data_size // test_batch_size):
                    test_loss_batch, test_acc_batch = sess.run([test_model.loss,
                                                                test_model.accuracy])
                    test_loss += test_loss_batch
                    test_acc += test_acc_batch

                test_loss = test_loss * test_batch_size / test_data_size
                test_acc = test_acc * test_batch_size / test_data_size


                result = 'global_step: {} | trn_loss : {} trn_acc : {} test loss : {} test acc : {}'.format(
                    global_step_val, np.mean(trn_loss),
                    np.mean(trn_acc), test_loss, test_acc)

                logger.info(result + '\n')
                logger.info(message + '\n')




                # l, acc = sess.run([test_model.loss, test_model.accuracy])
                # logger.info('reg epoch {} loss : {} accuracy {}'.format(global_step, l, acc))

                if test_loss < best_loss:
                    best_loss = test_loss
                    saver.save(sess, data_dir + '/model/' + model_dir + '/model.ckpt',
                               global_step=global_step_val)
                    patience = 0
                else:
                    patience += 1
                    if patience == max_patience:
                        logger.info('Max patience reached')
                        raise MaxPatienceError('Max patience')
                prev = time.time()

    except tf.errors.OutOfRangeError:
        logger.info('Done training -- epoch limit reached')
    except MaxPatienceError:
        logger.info('Done training -- max patience reached')
    finally:
        coord.request_stop()

    coord.join()
    sess.close()

