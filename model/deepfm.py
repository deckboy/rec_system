# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from config import config


def nn_tower(
        name, nn_input, hidden_units,
        activation=tf.nn.relu, use_bias=False,
        l2=0.0):
    out = nn_input
    for i, num in enumerate(hidden_units):
        out = tf.layers.dense(
            out,
            units=num,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2),
            use_bias=use_bias,
            activation=activation,
            name=name + "/layer_" + str(i),
        )
    return out


def deepfm_fn(inputs, is_test):
    # feature dimension：[batch, f_nums, weight_dim]
    weight = tf.reshape(
        inputs["feature_embedding"],
        shape=[-1, config['feature_len'], config['embedding_dim']])

    weight_ = tf.split(
        weight,
        num_or_size_splits=[config['embedding_dim'] - 1, 1],
        axis=2)

    # ================================================================
    #
    # linear part
    bias_part = tf.get_variable(
        "bias", [1, ],
        initializer=tf.zeros_initializer())  # 1*1

    linear_part = tf.nn.bias_add(
        tf.reduce_sum(weight_[1], axis=1),
        bias_part)  # batch*1

    # cross part
    # cross sub part : sum_square part
    summed_square = tf.square(tf.reduce_sum(weight_[0], axis=1))  #

    # batch*embed
    # cross sub part : square_sum part
    square_summed = tf.reduce_sum(tf.square(weight_[0]), axis=1)  # batch*embed
    cross_part = tf.subtract(summed_square, square_summed)

    feature_with_embedding_concat = tf.reshape(
        weight_[0],
        [-1, config['feature_len'] * (config['embedding_dim'] - 1)])

    dnn_out_ = nn_tower(
        'dnn_hidden',
        feature_with_embedding_concat, config['dnn_hidden_units'],
        use_bias=True, activation=config['activation_function'],
        l2=config['dnn_l2']
    )
    print(linear_part)
    print(cross_part)
    print(dnn_out_)
    out_ = tf.concat([linear_part, cross_part, dnn_out_], axis=1)
    print(out_)
    out_ = nn_tower('out', out_, [1], activation=None)

    out_tmp = tf.sigmoid(out_)  # batch
    if is_test:
        tf.add_to_collections("input_tensor", weight)
        tf.add_to_collections("output_tensor", out_tmp)

    # loss label = inputs["label"]  # [batch, 1]
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
        net_out_dic = deepfm_fn(inputs, is_test)

        loss = net_out_dic["loss"]

        result["out"] = net_out_dic
        if is_test:
            return result

        # ps - sgd
        emb_grad = tf.gradients(
            loss, [inputs["feature_embedding"]], name="feature_embedding")[0]

        result["feature_new_embedding"] = \
            inputs["feature_embedding"] - config['learning_rate'] * emb_grad

        result["feature_embedding"] = inputs["feature_embedding"]
        result["feature"] = inputs["feature"]

        # net - sgd
        tvars1 = tf.trainable_variables()
        grads1 = tf.gradients(loss, tvars1)
        opt = tf.train.GradientDescentOptimizer(
            learning_rate=config['learning_rate'],
            use_locking=True)
        train_op = opt.apply_gradients(zip(grads1, tvars1))
        result["train_op"] = train_op

        return result
