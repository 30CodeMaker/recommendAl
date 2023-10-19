import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import heapq
def top_k_heap(nums, k,items_items_sim):
    """
    :param nums: List
    :param k: int
    :return: List
    """
    heap = []
    for i in range(len(nums)):
        if len(heap) < k:
            heapq.heappush(heap, nums[i])
        else:
            if items_items_sim[nums[i]] > items_items_sim[heap[0]]:
                heapq.heappop(heap)
                heapq.heappush(heap, nums[i])
    return heap

def IOCCF_predict(user_items, items_items_sim, i, j, item_K_Neibors):
    Nj = set(item_K_Neibors[j]) & user_items[i]
    if len(Nj) == 0:
        return 0
    rui = 0.0
    for item in Nj:
        rui += items_items_sim[item][j]
    return rui

# 基于物品的协同过滤算法
def IOCCF(user_items, item_users, K, user_number, item_number, recommend_number, testData_users, testData_user_items,
          items,user_item_rating_prediction):

    items_items_sim = np.zeros((item_number + 1, item_number + 1))
    for i in range(1, item_number + 1):
        item_users.setdefault(i, set())

    for i in range(1, item_number + 1):
        for j in range(i + 1, item_number + 1):
            if len(item_users[i] | item_users[j]) != 0:
                items_items_sim[i][j] = len(item_users[i] & item_users[j]) / len(item_users[i] | item_users[j])
                items_items_sim[j][i] = items_items_sim[i][j]

    item_K_Neibors = dict()
    for i in range(1,item_number+1):
        item_K_Neibors.setdefault(i, set(range(1,item_number+1)))
        item_K_Neibors[i] = sorted(item_K_Neibors[i], key=lambda x: items_items_sim[i][x], reverse=True)[0:K]
    # for i in range(1, user_number + 1):
    #     user_items.setdefault(i, set())
    #     for j in range(1, item_number + 1):
    #         if j in user_items[i]:
    #             continue
    #         else:
    #             user_item_rating_prediction[i][j] = IOCCF_predict(user_items, item_users, items_items_sim, i, j, K)

    # 计算精确率
    Pre_K = 0.0
    Rec_K = 0.0

    for i in tqdm(range(len(testData_users))):
        user = testData_users[i]
        diff = list(items - user_items[user])
        for item in diff:
                user_item_rating_prediction[user][item] = IOCCF_predict(user_items, items_items_sim, user, item, item_K_Neibors)
        diff = sorted(diff, key=lambda x: user_item_rating_prediction[user][x], reverse=True)[0:recommend_number]
        Pre_K += len(set(diff) & testData_user_items[user]) / recommend_number
        Rec_K += len(set(diff) & testData_user_items[user]) / len(testData_user_items[user])
    Pre_K /= len(testData_users)
    Rec_K /= len(testData_users)
    print('IOCCF:')
    print(f'Pre@{recommend_number}:{Pre_K:.4f}')
    print(f'Rec@{recommend_number}:{Rec_K:.4f}')


def UOCCF_predict(item_users, user_user_sim, i, j, user_K_Neibors):
    Nu = set(user_K_Neibors[i]) & item_users[j]
    if len(Nu) == 0:
        return 0
    rui = 0.0
    for user in Nu:
        rui += user_user_sim[user][i]
    return rui

# UOCCF 基于用户的协同过滤算法
def UOCCF(user_items, item_users, K, user_number, item_number, recommend_number, testData_users, testData_user_items,
          items,user_item_rating_prediction):
    user_user_sim = np.zeros((user_number + 1, user_number + 1))
    for i in range(1,user_number+1):
        user_items.setdefault(i, set())

    for i in range(1, item_number + 1):
        item_users.setdefault(i, set())

    for i in range(1, user_number + 1):
        for j in range(i + 1, user_number + 1):
            if len(user_items[i] | user_items[j]) != 0:
                user_user_sim[i][j] = len(user_items[i] & user_items[j]) / len(user_items[i] | user_items[j])
                user_user_sim[j][i] = user_user_sim[i][j]

    user_K_Neibors = dict()
    for i in range(1,user_number+1):
        user_K_Neibors.setdefault(i, set(range(1,user_number+1)))
        user_K_Neibors[i] = sorted(user_K_Neibors[i], key=lambda x: user_user_sim[i][x], reverse=True)[0:K]

    # 计算精确率
    Pre_K = 0.0
    Rec_K = 0.0

    for i in tqdm(range(len(testData_users))):
        user = testData_users[i]
        diff = list(items - user_items[user])
        for item in diff:
                user_item_rating_prediction[user][item] = UOCCF_predict(item_users, user_user_sim, user, item, user_K_Neibors)
        diff = sorted(diff, key=lambda x: user_item_rating_prediction[user][x], reverse=True)[0:recommend_number]
        Pre_K += len(set(diff) & testData_user_items[user]) / recommend_number
        Rec_K += len(set(diff) & testData_user_items[user]) / len(testData_user_items[user])
    Pre_K /= len(testData_users)
    Rec_K /= len(testData_users)
    print('UOCCF:')
    print(f'Pre@{recommend_number}:{Pre_K:.4f}')
    print(f'Rec@{recommend_number}:{Rec_K:.4f}')

def HybridOCCF(IOCCF_user_item_rating_prediction,UOCCF_user_item_rating_prediction,items,testData_users,testData_user_items,recommend_number,a=0.5):
    Pre_K =0.0
    Rec_K = 0.0
    for user in testData_users:
        diff = list(items - user_items[user])
        for item in diff:
            IOCCF_user_item_rating_prediction[user][item] = IOCCF_user_item_rating_prediction[user][item]*0.5 + UOCCF_user_item_rating_prediction[user][item]*(1-a)
        diff = sorted(diff, key=lambda x: IOCCF_user_item_rating_prediction[user][x], reverse=True)[0:recommend_number]
        Pre_K += len(set(diff) & testData_user_items[user]) / recommend_number
        Rec_K += len(set(diff) & testData_user_items[user]) / len(testData_user_items[user])
    Pre_K /= len(testData_users)
    Rec_K /= len(testData_users)
    print('HybridOCCF:')
    print(f'Pre@{recommend_number}:{Pre_K:.4f}')
    print(f'Rec@{recommend_number}:{Rec_K:.4f}')

if __name__ == '__main__':
    # IOCCF 基于物品的协同过滤算法
    # UOCCF 基于用户的协同过滤算法
    start_time = time.time()
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
    items = set()
    testData_users = set()
    K = 50
    recommend_number = 5

    IOCCF_user_item_rating_prediction = np.zeros((user_number + 1, item_number + 1))
    UOCCF_user_item_rating_prediction = np.zeros((user_number + 1, item_number + 1))

    for index, row in u1_base.iterrows():
        if row['rating'] > 3:
            items.add(row['item_id'])
            user_items.setdefault(row['user_id'], set())
            user_items[row['user_id']].add(row['item_id'])
            item_users.setdefault(row['item_id'], set())
            item_users[row['item_id']].add(row['user_id'])

    for index, row in u1_test.iterrows():
        if row['rating'] > 3:
            items.add(row['item_id'])
            testData_user_items.setdefault(row['user_id'], set())
            testData_user_items[row['user_id']].add(row['item_id'])
            testData_item_users.setdefault(row['item_id'], set())
            testData_item_users[row['item_id']].add(row['user_id'])
            testData_users.add(row['user_id'])

    testData_users  = list(testData_users)
    IOCCF(user_items, item_users, K, user_number, item_number, recommend_number, testData_users, testData_user_items,
          items,IOCCF_user_item_rating_prediction)
    UOCCF(user_items, item_users, K, user_number, item_number, recommend_number, testData_users, testData_user_items,
          items,UOCCF_user_item_rating_prediction)
    HybridOCCF(IOCCF_user_item_rating_prediction,UOCCF_user_item_rating_prediction,items,testData_users,testData_user_items,recommend_number)
    end_time = time.time()
    print('cost %f seconds' % (end_time - start_time))