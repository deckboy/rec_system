# -*- coding: utf-8 -*-

from tqdm import tqdm
from sklearn import neighbors
import numpy as np
import pandas as pd
import warnings
import redis
import traceback
import json

warnings.filterwarnings('ignore')


def save_redis(items, db=1):
    redis_url = 'redis://:123456@127.0.0.1:6379/' + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()



def read_embedding_file(file):
    dic = dict()
    with open(file) as f:
        for line in f:
            tmp = line.split("\t")
            embedding = [float(_) for _ in tmp[1].split(",")]
            dic[int(tmp[0])] = embedding
    return dic


def embedding_sim(item_emb_file, cut_off=20):

    item_embedding = read_embedding_file(item_emb_file)
    item_idx_2_rawid_dict = {}
    item_emb_np = []
    for i, (k, v) in enumerate(item_embedding.items()):
        item_idx_2_rawid_dict[i] = k
        item_emb_np.append(v)

    item_emb_np = np.asarray(item_emb_np)

    item_emb_np = item_emb_np / np.linalg.norm(
        item_emb_np, axis=1, keepdims=True)

    # faiss/BallTree
    print("start build tree ... ")
    item_tree = neighbors.BallTree(item_emb_np, leaf_size=40)
    print("build tree end")
    sim, idx = item_tree.query(item_emb_np, cut_off)

    item_emb_sim_dict = {}
    for target_idx, sim_value_list, rele_idx_list in tqdm(
            zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        sim_tmp = {}
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            sim_tmp[rele_raw_id] = sim_value
        item_emb_sim_dict[target_raw_id] = sorted(
            sim_tmp.items(), key=lambda _: _[1], reverse=True)[:cut_off]

    print("start saved ...")
    item_simi_tuple = [(_, json.dumps(v)) for _, v in item_emb_sim_dict.items()]
    save_redis(item_simi_tuple, db=3)
    print("saved end")


if __name__ == '__main__':
    data_path = "../data/"
    item_emb_file = data_path + 'matrixcf_articles_emb.csv'
    embedding_sim(item_emb_file, 20)
