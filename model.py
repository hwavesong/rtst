# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class CompressModel(tf.keras.Model):
    def __init__(self):
        super(CompressModel, self).__init__()

        w_scale = np.sqrt(6 / (300 + 128))

        self.w = tf.Variable(tf.random.uniform([300, 128], -w_scale, w_scale, tf.float32, seed=37), trainable=True, dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([128]), trainable=True, dtype=tf.float32)

    def call(self, vec):
        return tf.tanh(tf.add(tf.matmul(vec, self.w), self.b))


def calculate_angle(edge1, edge2):
    raw_edge1_square = tf.reduce_sum(edge1 * edge1, axis=-1)
    raw_edge2_square = tf.reduce_sum(edge2 * edge2, axis=-1)

    clip_edge1_square = tf.clip_by_value(raw_edge1_square, 1e-8, tf.reduce_max(raw_edge1_square))  # avoid dividing zero by clipping value
    clip_edge2_square = tf.clip_by_value(raw_edge2_square, 1e-8, tf.reduce_max(raw_edge2_square))

    return tf.reduce_sum(edge1 * edge2, axis=-1) / tf.sqrt(clip_edge1_square * clip_edge2_square)


def calculate_RTST_A_loss(raw_A, A_hat):
    return tf.reduce_mean(tf.losses.mean_squared_error(raw_A, A_hat))


def calculate_RTST_BC_loss(raw_AB, raw_AC, short_AB, short_AC):
    raw_BC = raw_AB - raw_AC
    raw_B = calculate_angle(raw_BC, raw_AB)
    raw_C = calculate_angle(raw_BC, raw_AC)

    short_BC = short_AB - short_AC
    short_B = calculate_angle(short_BC, short_AB)
    short_C = calculate_angle(short_BC, short_AC)

    return (tf.reduce_mean(tf.losses.mean_squared_error(raw_B, short_B)) + tf.reduce_mean(tf.losses.mean_squared_error(raw_C, short_C))) / 2.


def calculate_Bin_loss(short_AB, short_AC):
    return (tf.reduce_mean(tf.square(short_AB)) + tf.reduce_mean(tf.square(short_AC))) / 2.


def get_train_op():
    return tf.optimizers.SGD(0.01)
