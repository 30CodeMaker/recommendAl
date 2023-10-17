import numpy as np
import pandas as pd
import time
import progressbar
# ALS算法
def ALS(A,d,t):
    p = progressbar.ProgressBar()
    U = np.random.rand(A.shape[0],d)
    V= np.random.rand(A.shape[1],d)
    u_row = len(U)
    v_row = len(V)
    #设置正则化项上的权衡参数
    au=0.01
    av=0.01

    #初始化用户的潜在特征向量
    for i in range(u_row):
        for j in range(d):
            U[i][j] = (U[i][j]-0.5)*0.01
    #初始化物品的潜在特征向量
    for i in range(v_row):
        for j in range(d):
            V[i][j] = (V[i][j]-0.5)*0.01
    #ALS算法
    for i in range(t):
        print(i)
        # 更新用户的潜在特征向量
        for j in range(1,u_row):
            Au = np.zeros((d,d))
            bu = np.zeros((1,d))
            for k in range(1,v_row):
                if A[j][k] != 0:
                    Au += (np.dot(V[k].reshape(d,1),V[k].reshape(1,d)) + au*np.eye(d))
                    bu += A[j][k]*V[k]
            #判断Au是否为奇异矩阵
            if np.linalg.det(Au) != 0:
                U[j] = np.dot(bu,np.linalg.inv(Au)).reshape(1,d)

        # 更新物品的潜在特征向量
        for j in range(1,v_row):
            bi = np.zeros((1,d))
            Ai = np.zeros((d,d))
            for k in range(1,u_row):
                if A[k][j] != 0:
                    Ai += (np.dot(U[k].reshape(d,1),U[k].reshape(1,d)) + av*np.eye(d))
                    bi += A[k][j]*U[k]
            #判断Ai是否为奇异矩阵
            if np.linalg.det(Ai) != 0:
                V[j] = np.dot(bi,np.linalg.inv(Ai)).reshape(1,d)

    #计算预测评分
    mae = 0.0
    rsme = 0.0
    u1_test = pd.read_csv('input/ml-100k/u1.test', sep='\t', header=None
                          , names=['user_id', 'item_id', 'rating', 'timestamp'])
    for index,row in u1_test.iterrows():
        r_prediction = np.dot(U[row['user_id']],V[row['item_id']].T)
        r_prediction = min(5, r_prediction)
        r_prediction = max(1, r_prediction)
        error = r_prediction - row['rating']
        mae += abs(error)
        rsme += (error) ** 2
    mae /= len(u1_test)
    rsme /= len(u1_test)
    rsme = np.sqrt(rsme)
    print(f"PMF_ALS(d={d},t={t}):")
    print('MAE =', mae)
    print('RSME =', rsme)

if __name__ == '__main__':
    userItem = np.zeros((944, 1683))
    u1_base = pd.read_csv('input/ml-100k/u1.base', sep='\t', header=None
                            , names=['user_id', 'item_id', 'rating', 'timestamp'])
    for index, row in u1_base.iterrows():
        userItem[row['user_id']][row['item_id']] = row['rating']
    start_time = time.time()
    ALS(userItem,20,100)
    end_time = time.time()
    print('cost %f seconds' % (end_time - start_time))