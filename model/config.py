# -*- coding: utf-8 -*-
import tensorflow as tf

config = {
    "feature_len": 8,
    "embedding_dim": 5,
    "label_len": 1,
    "n_parse_threads": 4,
    "shuffle_buffer_size": 1024,
    "prefetch_buffer_size": 1,
    "batch": 16,
    "learning_rate": 0.01,

    "dnn_hidden_units": [32, 16],
    "activation_function": tf.nn.relu,
    "dnn_l2": 0.1,

    "train_file": "../data2/train",
    "test_file": "../data2/val",
    "saved_embedding": "../data2/saved_dnn_embedding",
    "max_steps": 10000,
    "train_log_iter": 1000,
    "test_show_step": 1000,
    "last_test_auc": 0.5,

    "saved_checkpoint": "checkpoint",
    "checkpoint_name": "deepfm",

    "saved_pb": "../data2/saved_model",

}
