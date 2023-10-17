import random
import numpy as np
import pandas as pd
import time

#SVD++
def SVDPP(userItem, d, t, ratingList, bu, bi, global_mean,user_rated_item):
    U = np.random.rand(userItem.shape[0], d)
    V = np.random.rand(userItem.shape[1], d)
    U_ = np.zeros(d)
    W = np.random.rand(userItem.shape[1], d)
    U = np.array(U,dtype=np.float64)
    V = np.array(V,dtype=np.float64)
    u_row = len(U)
    v_row = len(V)
    # 设置正则化项上的权衡参数
    av = 0.01
    au = 0.01
    aw = 0.01
    betau = 0.01
    betav = 0.01
    # learning rate
    r = 0.01

    recordsLen = len(ratingList)
    # 初始化用户的潜在特征向量
    for i in range(u_row):
        for j in range(d):
            U[i][j] = (U[i][j] - 0.5) * 0.01
    # 初始化物品的潜在特征向量
    for i in range(v_row):
        for j in range(d):
            W[i][j] = (W[i][j] - 0.5) * 0.01
            V[i][j] = (V[i][j] - 0.5) * 0.01


    # SVDPP算法
    for i in range(t):
        print(i)
        for j in range(recordsLen):
            random_number = random.randint(0, recordsLen - 1)
            # 初始化用户虚拟属性的矩阵U_
            U_.fill(0)#清零

            rating_count = len(user_rated_item[ratingList[random_number][0]]-{ratingList[random_number][1]})
            sqrt_rating_count = np.sqrt(rating_count)

            if rating_count != 0:
                for item_id in user_rated_item[ratingList[random_number][0]]:
                    if item_id != ratingList[random_number][1]:
                        U_ += W[item_id]
                U_ = U_ / sqrt_rating_count

            r_preiction = (np.dot(U[ratingList[random_number][0]], V[ratingList[random_number][1]].T)
                           + bu[ratingList[random_number][0]] + bi[ratingList[random_number][1]] + global_mean
                           + np.dot(U_, V[ratingList[random_number][1]].T))

            eui = ratingList[random_number][2] - r_preiction

            gradientU = au * U[ratingList[random_number][0]] - eui * V[ratingList[random_number][1]]
            gradientV = av * V[ratingList[random_number][1]] - eui * (U[ratingList[random_number][0]]+U_)

            gradient_bu = betau * bu[ratingList[random_number][0]] - eui
            gradient_bi = betav * bi[ratingList[random_number][1]] - eui

            gradient_global_mean = -eui

            U[ratingList[random_number][0]] -=  r * gradientU
            V[ratingList[random_number][1]] -=  r * gradientV
            bu[ratingList[random_number][0]] -= r * gradient_bu
            bi[ratingList[random_number][1]] -= r * gradient_bi
            global_mean -= r * gradient_global_mean

            if rating_count != 0:
                for item_id in user_rated_item[ratingList[random_number][0]]:
                    if item_id != ratingList[random_number][1]:
                        gradient_W = aw * W[item_id] - (eui / sqrt_rating_count) * V[ratingList[random_number][1]]
                        W[item_id] -= r * gradient_W
        r = r*0.9


    # 计算预测评分
    mae = 0.0
    rsme = 0.0
    u1_test = pd.read_csv('input/ml-100k/ua.test', sep='\t', header=None
                          , names=['user_id', 'item_id', 'rating', 'timestamp'])
    for index, row in u1_test.iterrows():

        U_.fill(0)  # 清零
        rating_count = len(user_rated_item[row['user_id']] - {row['item_id']})
        sqrt_rating_count = np.sqrt(rating_count)

        if rating_count != 0:
            for item_id in user_rated_item[row['user_id']]:
                if item_id != row['item_id']:
                    U_ += W[item_id]
            U_ = U_ / sqrt_rating_count

        r_prediction = np.dot(U[row['user_id']], V[row['item_id']].T)+bu[row['user_id']]+bi[row['item_id']]+global_mean+np.dot(U_,V[row['item_id']].T)
        r_prediction = min(5, r_prediction)
        r_prediction = max(1,r_prediction)
        error = r_prediction - row['rating']
        mae += abs(error)
        rsme += error ** 2
    mae = mae / len(u1_test)
    rsme = (rsme / len(u1_test)) ** 0.5
    print(f'SVD++(d={d},T={t}):')
    print('MAE:', mae)
    print('RMSE:', rsme)

if __name__ == '__main__':
    userItem = np.zeros((944, 1683))
    user_rated_item = dict()

    u1_base = pd.read_csv('input/ml-100k/ua.base', sep='\t', header=None
                          , names=['user_id', 'item_id', 'rating', 'timestamp'])

    ua_base = u1_base.sample(frac=0.5,replace=False)
    remain_ua_base = u1_base.drop(ua_base.index)


    #记录每个用户对每个物品的评分
    ratingList = []
    for index, row in ua_base.iterrows():
        userItem[row['user_id']][row['item_id']] = row['rating']
        ratingList.append([row['user_id'], row['item_id'], row['rating']])

    for index,row in remain_ua_base.iterrows():
        userItem[row['user_id']][row['item_id']] = row['rating']
        user_rated_item.setdefault(row['user_id'],set())
        user_rated_item[row['user_id']].add(int(row['item_id']))

    user_mean = u1_base.groupby('user_id')['rating'].mean()
    item_mean = u1_base.groupby('item_id')['rating'].mean()
    global_mean = u1_base['rating'].mean()
    # 记录每个用户的平均评分
    ru = np.zeros(944)
    # 记录每个物品的平均评分
    ri = np.zeros(1683)
    for index, value in user_mean.items():
        ru[index] = value
    for index, value in item_mean.items():
        ri[index] = value

    bu = np.zeros(944)
    bi = np.zeros(1683)
    #计算用户偏置
    for i in range(1,944):
        count = 0
        for j in range(1,1683):
            if userItem[i][j]!=0:
                count+=1
                bu[i] += (userItem[i][j] - ru[i])
        if count!=0:
            bu[i] = bu[i]/count
    #计算物品偏置
    for j in range(1,1683):
        count = 0
        for i in range(1,944):
            if userItem[i][j]!=0:
                count+=1
                bi[j] += (userItem[i][j] - ri[j])
        if count!=0:
            bi[j] = bi[j]/count

    start_time = time.time()
    SVDPP(userItem, 20, 100, ratingList, bu, bi, global_mean,user_rated_item)
    end_time = time.time()
    print('cost %f seconds' % (end_time - start_time))