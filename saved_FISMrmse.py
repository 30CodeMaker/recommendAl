import math
import random
import numpy as np
import pandas as pd
import time
from tqdm import tqdm


class FISM_rmse:
    def __init__(self, train_data_file, test_data_file, T=5, d=20, learning_rate=0.01, regularization=0.001, alpha=0.5,
                 p=3):
        # initialize the model parameters
        self.p = p
        self.T = T
        self.d = d
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.user_num = 943
        self.item_num = 1682
        self.items = set(range(1, self.item_num + 1))
        self.bi = np.zeros(self.item_num + 1)
        self.bu = np.zeros(self.user_num + 1)
        self.user_item_matrix = np.zeros((self.user_num + 1, self.item_num + 1))

        # load the data and process it
        u_train = pd.read_csv(train_data_file, sep='\t', header=None,
                              names=['user_id', 'item_id', 'rating', 'timestamp'])
        u_train = u_train[u_train['rating'] > 3]
        self.observed_records = []
        u_test = pd.read_csv(test_data_file, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
        self.train_user_items = {}
        self.train_item_users = {}
        count = 0
        for index, row in u_train.iterrows():
            count += 1
            self.user_item_matrix[row['user_id']][row['item_id']] = 1
            self.train_item_users.setdefault(row['item_id'], set())
            self.train_user_items.setdefault(row['user_id'], set())
            self.train_item_users[row['item_id']].add(row['user_id'])
            self.train_user_items[row['user_id']].add(row['item_id'])
            self.observed_records.append((row['user_id'], row['item_id'], 1))

        self.unobserved_records = []
        self.test_data_users = set()
        self.test_user_items = {}
        for index, row in u_test.iterrows():
            if row['rating'] > 3:
                self.test_user_items.setdefault(row['user_id'], set())
                self.test_user_items[row['user_id']].add(row['item_id'])
                self.test_data_users.add(row['user_id'])

        # compute the bias of each item
        miu = count / (self.item_num * self.user_num)
        for i in range(1, self.item_num + 1):
            self.train_item_users.setdefault(i, set())
            self.bi[i] = self.train_item_users[i].__len__() / self.user_num - miu

        for i in range(1, self.user_num + 1):
            self.train_user_items.setdefault(i, set())
            self.bu[i] = self.train_user_items[i].__len__() / self.item_num - miu

        for i in range(1, self.user_num + 1):
            for j in range(1, self.item_num + 1):
                if self.user_item_matrix[i][j] == 0:
                    self.unobserved_records.append((i, j, 0))

        # initialize the latent matrix
        self.V = np.random.rand(self.item_num + 1, self.d)
        self.W = np.random.rand(self.item_num + 1, self.d)

        self.V = (self.V - 0.5) * 0.01
        self.W = (self.W - 0.5) * 0.01

    def predict(self, user_id, item_id):
        U_ = np.zeros(self.d, dtype=float)
        diff = self.train_user_items[user_id] - {item_id}
        rating_count = len(diff)
        if rating_count <= 0:
            return self.bu[user_id] + self.bi[item_id], diff, U_
        for item in diff:
            U_ = U_ + self.W[item]
        U_ = U_ / math.pow(rating_count, self.alpha)
        return np.dot(U_, self.V[item_id]) + self.bu[user_id] + self.bi[item_id], diff, U_

    def train(self):
        unobserved_records_length = len(self.unobserved_records)
        sample_length = self.p * len(self.observed_records)
        for t in tqdm(range(self.T)):
            all_index = range(unobserved_records_length)
            sample_index = random.sample(all_index, sample_length)
            sample_set = set()
            for i in sample_index:
                sample_set.add(self.unobserved_records[i])
            total_set = sample_set | set(self.observed_records)
            for record in total_set:
                user_id = record[0]
                item_id = record[1]

                r_prediction ,diff, U_ = self.predict(user_id,item_id)

                if len(diff) <= 0:
                    continue
                # U_ = np.zeros(self.d, dtype=float)
                # for item in diff:
                #     U_ = U_ + self.W[item]
                # U_ = U_ / math.pow(len(diff), self.alpha)

                # r_prediction = self.bu[user_id] + self.bi[item_id] + np.dot(U_, self.V[item_id])
                eui = record[2] - r_prediction

                self.W[list(diff)] -= self.learning_rate * (
                        self.regularization * self.W[list(diff)] - eui * math.pow(len(diff), -self.alpha) * self.V[
                    item_id])

                gradient_bu = self.regularization * self.bu[user_id] - eui
                gradient_V = self.regularization * self.V[item_id] - eui * U_
                gradient_bi = self.regularization * self.bi[item_id] - eui

                self.bi[item_id] -= self.learning_rate * gradient_bi
                self.bu[user_id] -= self.learning_rate * gradient_bu
                self.V[item_id] -= gradient_V * self.learning_rate


    def test(self, recommend_num=5):
        Pre_K = 0.0
        Rec_K = 0.0
        # compute the precision and recall of the model on test data set while the recommendation list length is 5
        for user in self.test_data_users:
            diff = self.items - self.train_user_items[user]
            user_item_rating_prediction = np.zeros(self.item_num + 1)
            for item in diff:
                user_item_rating_prediction[item] = self.predict(user,item)[0]

            diff = set(sorted(diff, key=lambda x: user_item_rating_prediction[x], reverse=True)[0:recommend_num])
            Pre_K += len(diff & self.test_user_items.get(user, set())) / recommend_num
            Rec_K += len(diff & self.test_user_items.get(user, set())) / len(self.test_user_items.get(user, set()))
        Pre_K /= len(self.test_data_users)
        Rec_K /= len(self.test_data_users)
        print(f'Pre@{recommend_num}:{Pre_K:.4f}')
        print(f'Rec@{recommend_num}:{Rec_K:.4f}')


if __name__ == '__main__':
    start = time.time()
    FISM = FISM_rmse('input/ml-100k/u1.base', 'input/ml-100k/u1.test')
    FISM.train()
    FISM.test()
    end = time.time()
    print(f'Running time:{end - start:.2f}s')
