import gzip
import numpy as np
from collections import defaultdict
import pandas as pd
from pandas.core.frame import DataFrame
import tqdm
import json

"""
Tool function for generating 5-core dataset
"""

def parse(path): # for Amazon
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
        
def Yelp(date_min, date_max, rating_score):
    users = []
    items = []
    scores = []
    times = []
    data_flie = './data_processing/Data/Yelp/yelp_academic_dataset_review_2020.json'
    lines = open(data_flie).readlines()
    for line in tqdm.tqdm(lines):
        review = json.loads(line.strip())
        rating = review['stars']
        # 2004-10-12 10:13:32 2019-12-13 15:51:19
        date = review['date']
        # 剔除一些例子
        if date < date_min or date > date_max or float(rating) <= rating_score:
            continue
        user = review['user_id']
        item = review['business_id']
        time = date.replace('-','').replace(':','').replace(' ','')
        users.append(user)
        items.append(item)
        scores.append(rating)
        times.append(time)
    return users,items,scores,times

# return (user item timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    users = []
    items = []
    scores = []
    times = []
    # older Amazon
    data_flie = './data_processing/Data/'+ dataset_name +'/reviews_' + dataset_name + '_5' + '.json.gz'
    # latest Amazon
    # data_flie = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    for inter in parse(data_flie):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        score = inter["overall"]
        time = inter['unixReviewTime']
        users.append(user)
        items.append(item)
        scores.append(score)
        times.append(time)
    return users,items,scores,times

# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    pop_users = set()
    pop_items = set()
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # 直接把user 删除
                user_items.pop(user)
                pop_users.add(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
                        pop_items.add(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items, pop_users, pop_items

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items: # 统计出现的次数
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

def id_map(user_items): # user_items dict

    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps

def get_interaction(datas):
    user_seq = {}
    for index, inter in datas.iterrows():
        user, item, time = inter['userId'],inter['itemId'],inter["timestamp"]
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

def main(data_name, data_type='Amazon'):
    assert data_type in {'Amazon', 'Yelp'}
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    if data_type == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2019-01-01 00:00:00'
        users, items, scores, times = Yelp(date_min, date_max, rating_score)
    else:
        users, items, scores, times = Amazon(data_name, rating_score=rating_score)

    data = DataFrame({
        "userId":users,
        "itemId":items,
        "rating":scores,
        "timestamp":times
    })
    
    user_items = get_interaction(data)
    user_items, pop_users, pop_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    
    
    data = data[(-data['userId'].isin(pop_users))]
    data = data[(-data['itemId'].isin(pop_items))]

    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id
    
    user2id = data_maps['user2id']
    item2id = data_maps['item2id']
    data['userId'] = data.userId.apply(lambda x: user2id[x])
    data['itemId'] = data.itemId.apply(lambda x: item2id[x])


    data.to_csv("./"+ data_name +".csv",index=False)
    
if __name__ =="__main__":
    main("Beauty", data_type="Amazon")