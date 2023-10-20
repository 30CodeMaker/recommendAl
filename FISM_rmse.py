import math
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import time

def FISM_rmse(test_data_users,test_user_items,user_items,bi,bu,observed_records,unobserved_records,T=100,d=20,learning_rate=0.01,regularization=0.001,alpha=0.5,p=3,recommend_num=5):
    items = set(range(1, 1683))
    item_num = len(items)
    V = np.random.rand(item_num+1, d)
    V = np.array(V, dtype=np.float64)
    W = np.random.rand(item_num+1, d)
    W = np.array(W, dtype=np.float64)
    v_row = len(V)

    # 初始化物品的潜在特征向量
    for i in range(v_row):
        for j in range(d):
            W[i][j] = (W[i][j] - 0.5) * 0.01
            V[i][j] = (V[i][j] - 0.5) * 0.01

    unobserved_records_length = len(unobserved_records)
    sample_length = len(observed_records) * p
    observed_records_set = set(observed_records)

    for t in tqdm(range(T)):
        all_index = range(unobserved_records_length)
        sample_index = random.sample(all_index, sample_length)
        sample_set = set()
        for i in range(sample_length):
            sample_set.add(unobserved_records[sample_index[i]])
        total_set = sample_set | observed_records_set

        for record in total_set:
            user_id =record[0]
            item_id = record[1]
            U_ = np.zeros(d, dtype=float)
            diff = user_items[user_id] - {item_id}
            rating_count = len(diff)
            if rating_count == 0:
                continue
            for item in diff:
                U_ += W[item]
            fm = math.pow(rating_count, alpha)
            U_ = U_ / fm
            r_prediction = np.dot(U_, V[item_id]) + bu[user_id] + bi[item_id]
            eui = record[2] - r_prediction

            W[list(diff)] -= learning_rate * (regularization * W[list(diff)] - (eui / fm) * V[list(diff)])

            gradient_V = regularization * V[item_id] - eui * U_
            gradient_bu = regularization * bu[user_id] - eui
            gradient_bi = regularization * bi[item_id] - eui

            V[item_id] -= learning_rate * gradient_V
            bu[user_id] -= learning_rate * gradient_bu
            bi[item_id] -= learning_rate * gradient_bi

    Pre_K = 0.0
    Rec_K = 0.0

    # compute the precision and recall of the model on test data set while the recommendation list length is 5
    for user in test_data_users:
        diff = items - user_items[user]
        user_item_rating_prediction = np.zeros(item_num+1)
        for item in diff:
            U_ = np.zeros(d)
            fm_diff = user_items[user] - {item}
            rating_count = len(fm_diff)
            if rating_count == 0:
                user_item_rating_prediction[item] = bu[user] + bi[item]
                continue
            for item_fm in fm_diff:
                U_ += W[item_fm]
            fm = math.pow(rating_count, alpha)
            U_ = U_ / fm
            user_item_rating_prediction[item] = bu[user] + bi[item] + np.dot(U_, V[item])
        diff = set(sorted(diff, key=lambda x: user_item_rating_prediction[x], reverse=True)[0:recommend_num])
        Pre_K += len(diff & test_user_items.get(user, set())) / recommend_num
        Rec_K += len(diff & test_user_items.get(user, set())) / len(test_user_items.get(user, set()))
    Pre_K /= len(test_data_users)
    Rec_K /= len(test_data_users)
    print(f'Pre@{recommend_num}:{Pre_K:.4f}')
    print(f'Rec@{recommend_num}:{Rec_K:.4f}')



if __name__ == '__main__':
    userItem = np.zeros((944, 1683))
    user_items = [set() for i in range(944)]
    u1_base = pd.read_csv('input/ml-100k/u1.base', sep='\t', header=None
                          , names=['user_id', 'item_id', 'rating', 'timestamp'])
    u1_test = pd.read_csv('input/ml-100k/u1.test', sep='\t', header=None
                          , names=['user_id', 'item_id', 'rating', 'timestamp'])
    #记录每个用户对每个物品的评分
    observed_records = []
    unobserved_records = []
    miu = 0
    test_data_users = set()
    test_user_items = {}
    for index, row in u1_base.iterrows():
        if row['rating'] > 3:
            miu += 1
            user_items[row['user_id']].add(row['item_id'])
            userItem[row['user_id']][row['item_id']] = 1
            observed_records.append((row['user_id'], row['item_id'], 1))
    for index,row in u1_test.iterrows():
        if row['rating'] > 3 :
            test_data_users.add(row['user_id'])
            test_user_items.setdefault(row['user_id'],set())
            test_user_items[row['user_id']].add(row['item_id'])

    miu = float(miu)/ (1682 * 943)
    bu = np.zeros(944)
    bi = np.zeros(1683)
    #计算用户偏置
    for i in range(1,944):
        count = 0
        for j in range(1,1683):
            if userItem[i][j]!=0:
                count+=1
            else:
                unobserved_records.append((i, j, 0))
        bu[i] = count/1682 - miu

    #计算物品偏置
    for j in range(1,1683):
        count = 0
        for i in range(1,944):
            if userItem[i][j]!=0:
                count+=1
        bi[j] = count/943 - miu

    start_time = time.time()
    FISM_rmse(test_data_users,test_user_items,user_items,bi,bu,observed_records,unobserved_records,10)
    end_time = time.time()
    print('cost %f seconds' % (end_time - start_time))