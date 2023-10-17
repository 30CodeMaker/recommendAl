import math

import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
    # PopRank算法实现
    u1_base = pd.read_csv('input/ml-100k/u1.base', header=None, sep='\t',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    u1_test = pd.read_csv('input/ml-100k/u1.test', header=None, sep='\t',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    user_number = 943
    item_number = 1682
    user_items = dict()
    item_users = dict()
    testData_user_items = dict()
    testData_item_users = dict()
    testData_users = set()
    K = 5

    user_item_rating_martix = np.zeros((user_number + 1, item_number + 1))

    # 预测值
    bi = np.zeros(item_number + 1)
    count = 0
    items = set()
    for index, row in u1_base.iterrows():
        items.add(row['item_id'])
        if row['rating'] > 3:
            count += 1
            user_items.setdefault(row['user_id'], set())
            user_items[row['user_id']].add(row['item_id'])
            item_users.setdefault(row['item_id'], set())
            item_users[row['item_id']].add(row['user_id'])

    for index, row in u1_test.iterrows():
        items.add(row['item_id'])
        if row['rating'] > 3:
            testData_user_items.setdefault(row['user_id'], set())
            testData_user_items[row['user_id']].add(row['item_id'])
            testData_item_users.setdefault(row['item_id'], set())
            testData_item_users[row['item_id']].add(row['user_id'])
            testData_users.add(row['user_id'])

    miu = count / (user_number * item_number)

    for i in range(item_number + 1):
        if i == 0:
            continue
        item_users.setdefault(i, set())
        bi[i] = len(item_users[i]) / user_number - miu

    # 计算精确率
    Pre_K = 0.0
    # 计算召回率
    Rec_K = 0.0
    # 计算F1分数
    F1_K = 0.0
    # 计算归一化折损累积增益
    NDCG_K = 0.0
    # 计算1-call值
    one_call_K = 0.0
    # 计算平均倒数排名
    MRR = 0.0
    # 计算平均精度均值
    MAP = 0.0
    # 计算平均相对位置
    ARP = 0.0
    # 计算曲线下面积
    AUC = 0.0

    # 测试集的用户数量
    testData_users_number = len(testData_users)
    # PopRank算法实现
    for user in testData_users:
        pre_count = 0

        DCG_Ku = 0.0
        Zu = 0.0
        min_location = 100000
        RPu = 0.0
        AUCu = 0.0

        diff = list(items - user_items[user])
        diff.sort(key=lambda x: bi[x], reverse=True)

        for i in range(K):
            l = i + 1
            if diff[i] in testData_user_items[user]:
                min_location = min(min_location, l)
                pre_count += 1
                DCG_Ku += 1 / math.log(l + 1, 2)

        times = 0
        if len(testData_user_items[user]) > 5:
            times = 5
        else:
            times = len(testData_user_items[user])
        for j in range(times):
            l = j + 1
            Zu += 1 / math.log(l + 1, 2)

        Pre_Ku = pre_count / K
        Rec_Ku = 0.0
        if len(testData_user_items[user]) != 0:
            Rec_Ku = pre_count / len(testData_user_items[user])

        Pre_K += Pre_Ku
        Rec_K += Rec_Ku
        if Pre_Ku + Rec_Ku != 0:
            F1_K += 2 * Pre_Ku * Rec_Ku / (Pre_Ku + Rec_Ku)
        NDCG_K += DCG_Ku / Zu
        if pre_count > 0:
            one_call_K += 1

        for item in diff:
            if item in testData_user_items[user]:
                l = diff.index(item) + 1
                min_location = min(min_location, l)
                break
        if min_location != 100000:
            MRR += 1 / min_location

        diff_len = len(diff)
        unlike_item_set = set(diff) - testData_user_items[user]
        for item in testData_user_items[user]:
            l = diff.index(item) + 1
            APu = 0.0
            for i in range(l):
                if diff[i] in testData_user_items[user]:
                    APu += 1
            APu /= l
            MAP += APu / len(testData_user_items[user])
            RPu += l / diff_len

            for unlike_item in unlike_item_set:
                if bi[item] > bi[unlike_item]:
                    AUCu += 1

        ARP += RPu/len(testData_user_items[user])

        AUC += AUCu/(len(testData_user_items[user])*len(unlike_item_set))



    print(f'Pre@{K}：{Pre_K / testData_users_number:.4f}')
    print(f'Rec@{K}：{Rec_K / testData_users_number:.4f}')
    print(f'F1@{K}：{F1_K / testData_users_number:.4f}')
    print(f'NDCG@{K}：{NDCG_K / testData_users_number:.4f}')
    print(f'1-call@{K}：{one_call_K / testData_users_number:.4f}')
    print(f'MRR：{MRR / testData_users_number:.4f}')
    print(f'MAP：{MAP / testData_users_number:.4f}')

    print(f'ARP：{ARP / testData_users_number:.4f}')
    print(f'AUC：{AUC / testData_users_number:.4f}')