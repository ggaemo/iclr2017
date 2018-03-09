import tensorflow as tf

class Discriminator:

    def __init__(self, partitioned_layers,mlp_struct):
        print('discrimiantor last discriminator disabled for unused calculation')
        self.loss = 0
        for idx, positive_input in enumerate(partitioned_layers[:-1]):
            negative_input = tf.concat(partitioned_layers[:idx] + partitioned_layers[(
                idx+1):],axis=0)
            inputs = tf.concat([positive_input, negative_input], axis=0)
            # positive_target = tf.gather(tf.ones_like(positive_input)., 0, axis=1)
            # negative_target = tf.gather(tf.zeros_like(negative_input), 0, axis=1)
            positive_target = tf.ones((positive_input.get_shape()[0], 1))
            negative_target = tf.zeros((negative_input.get_shape()[0], 1))
            target = tf.concat([positive_target, negative_target], axis=0)
            self.loss += self.mlp(inputs, target, mlp_struct)

    def mlp(self, inputs, target, mlp_struct):
        layer_list = list()
        layer_list.append(inputs)
        for layer_dim in mlp_struct:
            layer_input = layer_list[-1]
            layer = tf.contrib.layers.fully_connected(layer_input, layer_dim)
            layer_list.append(layer)
        output = tf.contrib.layers.fully_connected(layer_list[-1], 1, None)
        # target = tf.cast(target, tf.int32)
        # loss = tf.losses.sigmoid_cross_entropy(target, output)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                               logits=output))
        # self.output = tf.sigmoid(output)
        # self.target = target
        return loss






