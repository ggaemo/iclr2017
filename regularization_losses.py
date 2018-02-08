import itertools
import tensorflow as tf

class RegularizationLoss():

    def __init__(self, regularization_weight):
        self.regularization_weight = regularization_weight

class Dot(RegularizationLoss):

    def loss(self, layer_list):
        reg_loss = 0
        for i, j in itertools.combinations(layer_list, 2):
            normalize_i = tf.nn.l2_normalize(i, dim=1)
            normalize_j = tf.nn.l2_normalize(j, dim=1)
            reg_loss += tf.reduce_mean(tf.multiply(normalize_i, normalize_j))
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
            normalize_i = tf.nn.l2_normalize(i, dim=1)
            normalize_j = tf.nn.l2_normalize(j, dim=1)
            dot_prod = tf.reduce_sum(tf.multiply(normalize_i, normalize_j), axis=1)
            reg_loss += tf.losses.huber_loss(neg_target, dot_prod, self.delta)
        reg_loss = reg_loss * self.regularization_weight
        return reg_loss

class Orthogonal(RegularizationLoss):
    def loss(self, layer_list):
        reg_loss = 0
        for i, j in itertools.combinations(layer_list, 2):
            normalize_i = tf.nn.l2_normalize(i, dim=1)
            normalize_j = tf.nn.l2_normalize(j, dim=1)
            reg_loss += tf.abs(tf.reduce_sum(tf.multiply(normalize_i, normalize_j),
                                             axis=1))
        reg_loss = reg_loss * self.regularization_weight
        return reg_loss

class Hinge(RegularizationLoss):
    def loss(self, layer_list):
        reg_loss = 0
        neg_target = tf.gather(tf.ones_like(layer_list[0]), 0,
                               axis=1)
        for i, j in itertools.combinations(layer_list, 2):
            normalize_i = tf.nn.l2_normalize(i, dim=1)
            normalize_j = tf.nn.l2_normalize(j, dim=1)
            dot_prod = tf.reduce_sum(tf.multiply(normalize_i, normalize_j), axis=1)
            to_0_1 = (dot_prod + 1) / 2 # [0, 1] 사이에 들게 하기 위해서
            reg_loss += tf.losses.hinge_loss(neg_target, to_0_1)
        reg_loss = reg_loss * self.regularization_weight
        return reg_loss

class Dot_W(RegularizationLoss):

    def loss(self, weight):
        normalized = tf.nn.l2_normalize(weight, dim=0)
        weight = tf.matmul(tf.transpose(normalized), normalized)
        lower = tf.matrix_band_part(weight, -1, 0) # lower triangle
        reg_loss = tf.reduce_mean(lower)
        return reg_loss

class Orthogonal_W(RegularizationLoss):

    def loss(self, weight):
        normalized = tf.nn.l2_normalize(weight, dim=0)
        weight = tf.matmul(tf.transpose(normalized), normalized)
        lower = tf.matrix_band_part(weight, -1, 0) # lower triangle
        reg_loss = tf.reduce_mean(tf.abs(lower))
        return reg_loss
