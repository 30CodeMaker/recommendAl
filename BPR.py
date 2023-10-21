import random

import numpy as np
import pandas as pd
import time
from tqdm import tqdm

class BPR:
    def __init__(self, train_data_file, test_data_file,T=500,d=20,learning_rate=0.01,regularization=0.01):
        #initialize the model parameters
        self.T = T
        self.d = d
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.user_num = 943
        self.item_num = 1682
        self.items = set(range(1,self.item_num+1))
        self.bi = np.zeros(self.item_num+1)

        #load the data and process it
        u_train = pd.read_csv(train_data_file, sep='\t', header=None,names=['user_id', 'item_id', 'rating', 'timestamp'])
        u_train = u_train[u_train['rating'] > 3]
        self.user_item_pair = []
        u_test = pd.read_csv(test_data_file, sep='\t', header=None,names=['user_id', 'item_id', 'rating', 'timestamp'])
        self.train_user_items = {}
        self.train_item_users = {}
        count = 0
        for index,row in u_train.iterrows():
            count += 1
            self.train_item_users.setdefault(row['item_id'],set())
            self.train_user_items.setdefault(row['user_id'],set())
            self.train_item_users[row['item_id']].add(row['user_id'])
            self.train_user_items[row['user_id']].add(row['item_id'])
            self.user_item_pair.append((row['user_id'],row['item_id']))

        self.test_data_users = set()
        self.test_user_items = {}
        for index, row in u_test.iterrows():
            if row['rating'] > 3:
                self.test_user_items.setdefault(row['user_id'], set())
                self.test_user_items[row['user_id']].add(row['item_id'])
                self.test_data_users.add(row['user_id'])

        #compute the bias of each item
        miu = count / (self.item_num * self.user_num)
        for i in range(1,self.item_num+1):
            self.train_item_users.setdefault(i,set())
            self.bi[i] = self.train_item_users[i].__len__() / self.user_num - miu

        #initialize the latent matrix
        self.V = np.random.rand(self.item_num+1,self.d)
        self.U = np.random.rand(self.user_num+1,self.d)
        self.V = (self.V - 0.5) * 0.01
        self.U = (self.U - 0.5) * 0.01

        # self.rating_prediction_matrix = np.zeros((self.user_num+1,self.item_num+1))
        # for i in range(1,self.user_num+1):
        #     for j in range(1,self.item_num+1):
        #         self.rating_prediction_matrix[i][j] = self.predict(i,j)

    def predict(self,user_id,item_id):
        return np.dot(self.U[user_id],self.V[item_id]) + self.bi[item_id]

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def train(self):
        train_data_length = len(self.user_item_pair)
        for t in tqdm(range(self.T)):
            for i in range(train_data_length):
                # sample a user_item_pair
                user_id,item_i = self.user_item_pair[random.randint(0,train_data_length-1)]
                Iu = self.train_user_items.get(user_id, set())
                item_j = list(self.items - Iu)
                item_j = item_j[random.randint(0,len(item_j)-1)]

                #compute the gradient
                # x_uij = self.rating_prediction_matrix[user_id][item_i] - self.rating_prediction_matrix[user_id][item_j]
                x_uij = self.predict(user_id,item_i) - self.predict(user_id,item_j)
                sigm = self.sigmoid(-x_uij)
                gradient_U = self.regularization * self.U[user_id] - sigm * (self.V[item_i] - self.V[item_j])
                gradient_Vi = self.regularization * self.V[item_i] - sigm * self.U[user_id]
                gradient_Vj = self.regularization * self.V[item_j] + sigm * self.U[user_id]
                gradient_bi = self.regularization * self.bi[item_i] - sigm
                gradient_bj = self.regularization * self.bi[item_j] + sigm

                #update the parametersd
                self.U[user_id] -= self.learning_rate * gradient_U
                self.V[item_i] -= self.learning_rate * gradient_Vi
                self.V[item_j] -= self.learning_rate * gradient_Vj
                self.bi[item_i] -= self.learning_rate * gradient_bi
                self.bi[item_j] -= self.learning_rate * gradient_bj
    def test(self,recommend_num=5):
        Pre_K = 0.0
        Rec_K = 0.0
        #compute the precision and recall of the model on test data set while the recommendation list length is 5
        for user in self.test_data_users:
            diff = self.items - self.train_user_items[user]
            user_item_rating_prediction = np.zeros(self.item_num+1)
            for item in diff:
                user_item_rating_prediction[item] = self.predict(user,item)
            diff = set(sorted(diff, key=lambda x: user_item_rating_prediction[x], reverse=True)[0:recommend_num])
            Pre_K += len(diff & self.test_user_items.get(user,set())) / recommend_num
            Rec_K += len(diff & self.test_user_items.get(user,set())) / len(self.test_user_items.get(user,set()))
        Pre_K /= len(self.test_data_users)
        Rec_K /= len(self.test_data_users)
        print(f'Pre@{recommend_num}:{Pre_K:.4f}')
        print(f'Rec@{recommend_num}:{Rec_K:.4f}')

#Bayesian Personalized Ranking
if __name__ == '__main__':
    start = time.time()
    BPR = BPR('input/ml-100k/u1.base','input/ml-100k/u1.test')
    BPR.train()
    BPR.test()
    end = time.time()
    print(f'Running time:{end-start:.2f}s')