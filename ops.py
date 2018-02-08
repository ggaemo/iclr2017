

import tensorflow as tf


def fully_connected_bn(layer_input, num_outputs, is_training, activation_fn=tf.nn.relu):
    bn_layer = tf.contrib.layers.batch_norm(layer_input, is_training=is_training,
                                            fused=True, center=True, scale=True)
    layer = tf.contrib.layers.fully_connected(bn_layer, num_outputs, activation_fn,
                                              variables_collections=['W_b'])

    '''
    이런 방식으로 하는게 맞나..??? (is_training을 init에서 나온 scope에서 가져오독 하는것)
    '''
    return layer


def fully_connected(layer_input, num_outputs, activation_fn=tf.nn.relu):
    # layer = tf.contrib.layers.fully_connected(layer_input, num_outputs,
    #                                           activation_fn,
    #                                         variables_collections=['W_b'])

    layer = tf.contrib.layers.fully_connected(layer_input, num_outputs,
                                              activation_fn,
                                              variables_collections=['W_b'])
    return layer


def conv_block_encode(input, filters, kernel_size, strides, padding, is_training,
                      activation_fn=tf.nn.relu):
    conv_output = tf.contrib.layers.conv2d(input, filters, kernel_size,
                                                     strides, padding,
                                                     normalizer_fn=tf.contrib.layers.batch_norm,
                                                     normalizer_params={'fused':
                                                                            True,
                                                                        'scale': True,
                                                                        'is_training'
                                                                        : is_training
                                                                        },
                                                     activation_fn=activation_fn)

    # conv_output = tf.contrib.layers.conv2d(input, filters, kernel_size,
    #                                                  strides, padding,
    #                                                  activation_fn=tf.nn.relu)

    # conv_output = tf.contrib.layers.conv2d(input, filters, kernel_size, strides,
    #                                        padding, biases_initializer=None,
    #                                        activation_fn=None)
    # bn_output = tf.contrib.layers.batch_norm(conv_output, fused=True, scale=True,
    #                                           is_training=is_training)
    # relu_output = tf.nn.relu(bn_output)
    return conv_output

def conv_block_decode(input, filters, kernel_size, strides, padding, is_training,
                      activation_fn=tf.nn.relu):
    conv_output = tf.contrib.layers.conv2d_transpose(input, filters, kernel_size,
                                                     strides, padding,
                                                     normalizer_fn=tf.contrib.layers.batch_norm,
                                                     normalizer_params = {'fused' :
                                                                              True,
                                                                          'scale' : True,
                                                                          'is_training'
                                                                          : is_training
                                                                          },
                                                     activation_fn = activation_fn)

    # conv_output = tf.contrib.layers.conv2d_transpose(input, filters, kernel_size,
    #                                                  strides, padding,
    #                                                  activation_fn = tf.nn.relu)
    return conv_output

def conv_block_discrim(input, filters, kernel_size, strides, padding, alpha, is_training):
    conv_output = tf.contrib.layers.conv2d(input, filters, kernel_size, strides, padding)
    bn_output = tf.contrib.layers.batch_norm(conv_output, fused=True,
                                             is_training=is_training)
    lrelu_output = tf.maximum(alpha * bn_output, bn_output)
    return lrelu_output

def get_param(input, dim, deterministic):
    mean = tf.contrib.layers.fully_connected(input, dim, activation_fn=None)

    if deterministic:
        log_sigma_sq = tf.zeros_like(mean)
        value = mean
    else:
        log_sigma_sq = tf.contrib.layers.fully_connected(input, dim,
                                                         activation_fn=None)
        value = mean + tf.sqrt(tf.exp(log_sigma_sq)) * tf.random_normal(
            tf.shape(log_sigma_sq))
    return (mean, log_sigma_sq, value)


def make_partitioned_layer(layer_input, num_partition, num_outputs):
    partitioned_layer_list = list()
    for _ in range(num_partition):
        partitioned_layer_list.append(tf.contrib.layers.fully_connected(layer_input,
                                                                  num_outputs))
    return partitioned_layer_list