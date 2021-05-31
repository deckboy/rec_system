# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf

import struct
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import numpy as np

# slots = 18
pb_path = "../data2/saved_model"
feature_embeddings = "../data2/saved_dnn_embedding"


def load_embed():
    hashcode_dict = {}

    with open(feature_embeddings, "r") as lines:
        for line in lines:
            tmp = line.strip().split("\t")
            if len(tmp) == 2:
                vec = [float(_) for _ in tmp[1].split(",")]
                hashcode_dict[tmp[0]] = vec

    return hashcode_dict


def load_model():
    sess = tf.Session()
    # tf.saved_model.loader.load(sess, ['serve'], model_file)
    meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING],
                                                pb_path)
    signature = meta_graph_def.signature_def
    # get tensor name
    in_tensor_name = \
    signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
        'input0'].name
    out_tensor_name = \
    signature[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'output0'].name
    # get tensor
    in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
    out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)

    return sess, in_tensor, out_tensor
    #


def batch_predict():
    """
    input_data : rank by slotid, separate by \t

    """
    embed_dict = load_embed()
    sess, in_tensor, out_tensor = load_model()
    print(in_tensor)
    keys = embed_dict.keys()
    print("====load success====")
    batch_sample = []
    hashcode = []

    for _ in range(16):
        slic = random.sample(keys, 8)
        hashcode.append(",".join([str(_) for _ in slic]))
        hashcode_embed = []
        for s in slic:
            hashcode_embed.append(embed_dict[s])
        batch_sample.append(hashcode_embed)
    print("raw data: \n", np.array(batch_sample))
    prediction = sess.run(out_tensor, feed_dict={in_tensor: np.array(batch_sample)})
    for i, p in enumerate(prediction):
        print(hashcode[i] + "\t")
        print(str(p) + "\n")


def main():
    batch_predict()


if __name__ == '__main__':
    main()
