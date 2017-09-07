
import tensorflow as tf


class SimpleModel():
    def __init__(self, inputs, target, num_outputs_by_layer, is_training):
        def fully_connected_bn(layer_input, num_outputs, is_training):
            layer = tf.contrib.layers.fully_connected(layer_input, num_outputs)
            layer = tf.contrib.layers.batch_norm(layer, is_training = is_training)
            '''
            이런 방식으로 하는게 맞나..??? (is_training을 init에서 나온 scope에서 가져오독 하는것)
            '''
            return layer

        layer_list = list()
        layer_list.append(inputs)
        for num_outputs in num_outputs_by_layer:
            layer_list.append(tf.contrib.layers.fully_connected_bn(layer_list[-1],
                                                                num_outputs))


        output_layer = tf.contrib.layers.fully_connected(layer_list[-1], num_outputs = 10)

        self.loss = tf.losses.softmax_cross_entropy(target, output_layer)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        with tf.variable_scope('metrics'):
            correct_pred = tf.equal(tf.argmax(target, 1), tf.argmax(output_layer, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.histogram('layer_1', layer_1)

        # tf.summary.histogram('activation', tf.cast(tf.equal(layer_1, 0.0),tf.float32))
        zeros = tf.zeros_like(layer_1, dtype=tf.float32)
        tf.summary.histogram('activation',
                             tf.cast(tf.greater(layer_1, 0.0), tf.int32))

        tf.summary.histogram('output_layer', output_layer)
        tf.summary.scalar('xent_loss', self.loss)
        # tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)