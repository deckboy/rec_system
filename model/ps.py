# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


class Singleton(type):
    _instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instance:
            Singleton._instance[cls] = type.__call__(cls, *args, **kwargs)
        return Singleton._instance[cls]


# k-v map{hashcode, embedding}
class PS(metaclass=Singleton):
    def __init__(self, embedding_dim):
        np.random.seed(2020)
        self.params_server = dict()
        self.dim = embedding_dim
        print("ps inited...")

    def pull(self, keys):
        values = []
        # [batch, feature_len]
        for k in keys:
            tmp = []
            for arr in k:
                value = self.params_server.get(arr, None)
                if value is None:
                    value = np.random.rand(self.dim)
                    self.params_server[arr] = value
                tmp.append(value)
            values.append(tmp)

        return np.asarray(values, dtype='float32')

    def push(self, keys, values):
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                self.params_server[keys[i][j]] = values[i][j]

    def delete(self, keys):
        for k in keys:
            self.params_server.pop(k)

    def save(self, path):
        print("keys: ", len(self.params_server))
        writer = open(path, "w")
        for k, v in self.params_server.items():
            writer.write(
                str(k) + "\t" + ",".join(['%.8f' % _ for _ in v]) + "\n")
        writer.close()


if __name__ == '__main__':
    # testing ps

    ps_local = PS(8)
    keys = [123, 234]
    # pull keys
    res = ps_local.pull(keys)
    print(ps_local.params_server)
    print(res)

    # push to ps
    gradient = 10
    res = res - 0.01 * gradient
    ps_local.push(keys, res)
    print(ps_local.params_server)

    # save embedding
    path = "F:\\6_class\\first_project\\feature_embedding"
    ps_local.save(path)
