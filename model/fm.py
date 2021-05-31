# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

weight_dim = 17
learning_rate = 0.01
f_nums = 4


def fm_fn(inputs, is_test):
    # feature dimension：user_id 和 movie_id
    weight_ = tf.reshape(
        inputs["feature_embedding"],
        shape=[-1, f_nums, weight_dim])  # [batch, f_nums, weight_dim]

    # split linear weight and cross weight
    weight_ = tf.split(weight_, num_or_size_splits=[weight_dim - 1, 1], axis=2)

    # linear part
    bias_part = tf.get_variable(
        "bias", [1, ],
        initializer=tf.zeros_initializer())  # 1*1

    linear_part = tf.nn.bias_add(
        tf.reduce_sum(weight_[1], axis=1),
        bias_part)  # batch*1

    # cross part
    # cross sub part : sum_square part
    summed_square = tf.square(tf.reduce_sum(weight_[0], axis=1))  # batch*embed
    # cross sub part : square_sum part
    square_summed = tf.reduce_sum(tf.square(weight_[0]), axis=1)  # batch*embed
    cross_part = 0.5 * tf.reduce_sum(
        tf.subtract(summed_square, square_summed),
        axis=1, keepdims=True)  # batch*1
    out_ = linear_part + cross_part
    out_tmp = tf.sigmoid(out_)  # batch
    if is_test:
        tf.add_to_collections("input_tensor", weight_)
        tf.add_to_collections("output_tensor", out_tmp)

    # 损失函数loss label = inputs["label"]  # [batch, 1]
    loss_ = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=out_, labels=inputs["label"]))

    out_dic = {
        "loss": loss_,
        "ground_truth": inputs["label"][:, 0],
        "prediction": out_[:, 0]
    }

    return out_dic


# define graph
def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope("net_graph", reuse=is_test):
        # init graph
        net_out_dic = fm_fn(inputs, is_test)

        loss = net_out_dic["loss"]
        result["out"] = net_out_dic

        if is_test:
            return result

        # SGD
        emb_grad = tf.gradients(
            loss, [inputs["feature_embedding"]], name="feature_embedding")[0]

        result["feature_new_embedding"] = \
            inputs["feature_embedding"] - learning_rate * emb_grad

        result["feature_embedding"] = inputs["feature_embedding"]
        result["feature"] = inputs["feature"]

        return result
