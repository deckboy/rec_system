# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

batch = 32
embedding_dim = 8
learning_rate = 0.001


def mf_fn(inputs, is_test):
    # feature dimension：user_id 和 movie_id
    embed_layer = inputs["feature_embedding"]  # [batch, 2, embedding_dim]
    embed_layer = tf.reshape(embed_layer, shape=[-1, 2, embedding_dim])
    label = inputs["label"]  # [batch, 1]
    # user_id embedding and movie_id embedding
    embed_layer = tf.split(embed_layer, num_or_size_splits=2, axis=1)  # [batch, embedding_dim] * 2
    user_id_embedding = tf.reshape(embed_layer[0], shape=[-1, embedding_dim])  # [batch, embedding_dim]
    movie_id_embedding = tf.reshape(embed_layer[1], shape=[-1, embedding_dim])  # [batch, embedding_dim]
    # sum of product
    out_ = tf.reduce_mean(
        user_id_embedding * movie_id_embedding, axis=1)  # [batch]
    #
    out_tmp = tf.sigmoid(out_)  # batch
    if is_test:
        tf.add_to_collections("input_tensor", embed_layer)
        tf.add_to_collections("output_tensor", out_tmp)

    # loss
    label_ = tf.reshape(label, [-1])  # [batch]
    loss_ = tf.reduce_sum(tf.square(label_ - out_))  # 1

    out_dic = {
        "loss": loss_,
        "ground_truth": label_,
        "prediction": out_
    }

    return out_dic


# define graph
def setup_graph(inputs, is_test=False):
    result = {}
    with tf.variable_scope("net_graph", reuse=is_test):
        # initialize
        net_out_dic = mf_fn(inputs, is_test)

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
