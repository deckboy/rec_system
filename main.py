

import pandas as pd


ds = pd.read_csv("data/click_log.csv")
# a_liat = ds['user_id'].unique()
# a_liat = list(a_liat)[: 10000]
#
# ds01 = ds[ds['user_id'].isin(a_liat)]
# ds01.to_csv("data/click_log_1.csv", index=False, header=True)
a_liat = ds['article_id'].unique()
#
# ds01 = pd.read_csv("data/articles.csv")
# #
# ds01 = ds01[ds01['article_id'].isin(a_liat)]
# print(ds01)
# #
# ds01.to_csv("data/articles_1.csv", index=False, header=True)


ds01 = pd.read_csv("data/articles_emb.csv")
#
ds01 = ds01[ds01['article_id'].isin(a_liat)]
# print(ds01)
# #
ds01.to_csv("data/articles_emb_1.csv", index=False, header=True)


f = open('data/articles_emb_1.csv')

w = open('data/matrixcf_articles_emb.csv', 'w')
for i, line in enumerate(f):
    if i == 0:
        continue
    tmp = line.strip().split(',', 1)
    w.write(tmp[0] + "\t" + tmp[1] + "\n")

w.close()


# import pandas as pd
#
# ds = pd.read_csv('./data/train_tohash', names=['user', 'item', 'score'])
#
# print(len(ds['item'].unique()))
# print(len(ds['user'].unique()))

