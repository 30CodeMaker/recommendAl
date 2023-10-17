import random
import numpy as np
import pandas as pd
import time

def SGD(userItem, d, t, ratingList):
    U = np.random.rand(userItem.shape[0], d)
    V = np.random.rand(userItem.shape[1], d)
    U = np.array(U,dtype=np.float64)
    V = np.array(V,dtype=np.float64)
    u_row = len(U)
    v_row = len(V)
    # 设置正则化项上的权衡参数
    av = 0.01
    au = 0.01
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
            V[i][j] = (V[i][j] - 0.5) * 0.01
    # SGD算法
    for i in range(t):
        print(i)
        for j in range(recordsLen):
            random_number = random.randint(0, recordsLen - 1)

            gradientU = (U[ratingList[random_number][0]] * au
                         - (ratingList[random_number][2] - np.dot(U[ratingList[random_number][0]], V[ratingList[random_number][1]].T))
                         * V[ratingList[random_number][1]])
            gradientV = (V[ratingList[random_number][1]] * av
                         - (ratingList[random_number][2] - np.dot(U[ratingList[random_number][0]], V[ratingList[random_number][1]].T))
                         * U[ratingList[random_number][0]])
            U[ratingList[random_number][0]] -=  r * gradientU
            V[ratingList[random_number][1]] -=  r * gradientV
        r = r*0.9
    # 计算预测评分
    mae = 0.0
    rsme = 0.0
    u1_test = pd.read_csv('input/ml-100k/u1.test', sep='\t', header=None
                          , names=['user_id', 'item_id', 'rating', 'timestamp'])
    for index, row in u1_test.iterrows():
        r_prediction = np.dot(U[row['user_id']], V[row['item_id']].T)
        r_prediction = min(5, r_prediction)
        r_prediction = max(1,r_prediction)
        error = r_prediction - row['rating']
        mae += abs(error)
        rsme += error ** 2
    mae = mae / len(u1_test)
    rsme = (rsme / len(u1_test)) ** 0.5
    print(f'SGD(d={d},T={t}):')
    print('MAE:', mae)
    print('RMSE:', rsme)

if __name__ == '__main__':
    userItem = np.zeros((944, 1683))
    u1_base = pd.read_csv('input/ml-100k/u1.base',sep='\t',header=None
                          ,names=['user_id','item_id','rating','timestamp'])
    ratingList = []
    for index, row in u1_base.iterrows():
        userItem[row['user_id']][row['item_id']] = row['rating']
        ratingList.append([row['user_id'], row['item_id'], row['rating']])
    start_time = time.time()
    SGD(userItem, 20, 100, ratingList)
    end_time = time.time()
    print('cost %f seconds' % (end_time - start_time))