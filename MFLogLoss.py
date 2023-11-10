import random
import numpy as np
import pandas as pd
import time
from tqdm import tqdm


class MFLogLoss:
    def __init__(self, train_data_file, test_data_file, T=100, d=20, learning_rate=0.01, regularization=0.001, alpha=0.5,
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
                    self.unobserved_records.append((i, j, -1))

        # initialize the latent matrix
        self.V = np.random.rand(self.item_num + 1, self.d)
        self.U = np.random.rand(self.user_num + 1, self.d)

        self.V = (self.V - 0.5) * 0.01
        self.U = (self.U - 0.5) * 0.01

    def predict(self, user_id, item_id):
        return self.bu[user_id] + self.bi[item_id] + np.dot(self.U[user_id], self.V[item_id])

    def train(self):
        unobserved_records_length = len(self.unobserved_records)
        sample_length = len(self.observed_records) * self.p
        observed_records_set = set(self.observed_records)
        for t in tqdm(range(self.T)):
            all_index = range(unobserved_records_length)
            sample_index = random.sample(all_index, sample_length)
            sample_set = set()
            for i in sample_index:
                sample_set.add(self.unobserved_records[i])
            total_set = sample_set | observed_records_set
            for record in total_set:
                user_id = record[0]
                item_id = record[1]

                r_prediction= self.predict(user_id, item_id)
                eui = float(record[2]) / (1+np.exp(r_prediction*record[2]))

                gradient_U = self.regularization * self.U[user_id] - eui * self.V[item_id]
                gradient_V = self.regularization * self.V[item_id] - eui * self.U[user_id]
                gradient_bu = self.regularization * self.bu[user_id] - eui
                gradient_bi = self.regularization * self.bi[item_id] - eui

                self.U[user_id] -= self.learning_rate * gradient_U
                self.V[item_id] -= self.learning_rate * gradient_V
                self.bu[user_id] -= self.learning_rate * gradient_bu
                self.bi[item_id] -= self.learning_rate * gradient_bi
    def test(self, recommend_num=5):
        Pre_K = 0.0
        Rec_K = 0.0
        # compute the precision and recall of the model on test data set while the recommendation list length is 5
        for user in self.test_data_users:
            diff = self.items - self.train_user_items[user]
            user_item_rating_prediction = np.zeros(self.item_num + 1)
            for item in diff:
                user_item_rating_prediction[item]= self.predict(user, item)
            diff = set(sorted(diff, key=lambda x: user_item_rating_prediction[x], reverse=True)[0:recommend_num])
            Pre_K += len(diff & self.test_user_items.get(user, set())) / recommend_num
            Rec_K += len(diff & self.test_user_items.get(user, set())) / len(self.test_user_items.get(user, set()))
        Pre_K /= len(self.test_data_users)
        Rec_K /= len(self.test_data_users)
        print(f"MFLogLoss:")
        print(f'Pre@{recommend_num}:{Pre_K:.4f}')
        print(f'Rec@{recommend_num}:{Rec_K:.4f}')

if __name__ == "__main__":
    start = time.time()
    mfLogLoss = MFLogLoss('input/ml-100k/u4.base', 'input/ml-100k/u4.test')
    mfLogLoss.train()
    mfLogLoss.test()
    end = time.time()
    print(f'Running time:{end - start:.2f}s')