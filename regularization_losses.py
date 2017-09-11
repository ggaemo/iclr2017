import itertools
import tensorflow as tf

class RegularizationLoss():

    def __init__(self, regularization_weight):
        self.regularization_weight = regularization_weight

class Dot(RegularizationLoss):

    def loss(self, layer_list):
        reg_loss = 0
        for i, j in itertools.combinations(layer_list, 2):
            reg_loss += tf.reduce_mean(tf.multiply(i, j))
        reg_loss = reg_loss * self.regularization_weight
        return reg_loss

class DotHuber(RegularizationLoss):
    def __init__(self, reguarlization_weight, huber_target, delta):
        super().__init__(reguarlization_weight)
        self.huber_target, self.delta = huber_target, delta

    def loss(self, layer_list):
        reg_loss = 0
        # neg_ones = -tf.ones(layer_list.shape[0])
        neg_target = -tf.gather(tf.ones_like(layer_list[0]), 0, axis=1) * self.huber_target
        for i, j in itertools.combinations(layer_list, 2):
            dot_prod = tf.reduce_mean(tf.multiply(i, j), axis=1)
            reg_loss += tf.losses.huber_loss(neg_target, dot_prod, self.delta)
        reg_loss = reg_loss * self.regularization_weight
        return reg_loss

class Orthogonal(RegularizationLoss):
    def loss(self, layer_list):
        reg_loss = 0
        for i, j in itertools.combinations(layer_list, 2):
            reg_loss += tf.sqrt(tf.reduce_mean((tf.square(tf.multiply(i, j)))))
        reg_loss = reg_loss * self.regularization_weight
        return reg_loss