# import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import time
import csv

# User-based Collaborative Filtering
class UserBasedCF:
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = pd.DataFrame()
        self.trainData = {}
        self.testData = {}
        self.userSimMatrix = []
        self.user_mean = {}
        # 物品用户倒排表
        self.item_users = dict()
        # 用户物品倒排表
        self.user_items = dict()

    # read data from input_file
    def readData(self):
        self.data = pd.read_csv(self.datafile, sep='\t', header=None,
                                names=['user_id', 'item_id', 'rating', 'timestamp'])
        user_group = self.data.groupby('user_id')
        # meanUserScores
        self.user_mean = user_group['rating'].mean()

    def preprocessData(self):
        traindata_list = {}
        # 存储格式：
        for index, row in self.data.iterrows():
            traindata_list.setdefault(row['user_id'], {})
            traindata_list[row['user_id']][row['item_id']] = row['rating']
        self.trainData = traindata_list

    # 计算用户相似度
    def usersim(self, user1, user2):
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        # 求user1和user2评分物品的交集
        user_items_intersection = self.user_items[user1] & self.user_items[user2]
        if len(user_items_intersection) == 0:
            return 0
        # 遍历user1和user2评分物品的交集
        for item in user_items_intersection:
            sum_x += (self.trainData[user1][item] - self.user_mean[user1]) ** 2
            sum_y += (self.trainData[user2][item] - self.user_mean[user2]) ** 2
            sum_xy += (self.trainData[user1][item] - self.user_mean[user1]) * (
                        self.trainData[user2][item] - self.user_mean[user2])

        # ???? 如果分母为0，怎么办
        if (sum_x == 0 or sum_y == 0):
            return 0
        return sum_xy / ((sum_x * sum_y) ** 0.5)

    # 计算用户相似度矩阵
    def userSimilarity(self):
        user_count = len(self.user_mean)

        for u, item in self.trainData.items():
            self.user_items.setdefault(u, set())
            for i in item.keys():
                self.user_items[u].add(i)
                self.item_users.setdefault(i, set())
                self.item_users[i].add(u)
        self.userSimMatrix = np.zeros((user_count + 1, user_count + 1))

        # 计算用户相似度矩阵（皮尔逊相关系数）
        for i in range(1, user_count + 1):
            for j in range(i, user_count + 1):
                if i == j:
                    self.userSimMatrix[i][j] = 1
                else:
                    self.userSimMatrix[i][j] = self.userSimMatrix[j][i] = self.usersim(i, j)

    # 预测用户对物品的评分
    def ratingPrediction(self, user, item, k=8):
        Nu = set()
        user_count = len(self.user_mean)
        self.item_users.setdefault(item, set())
        for i in range(1, user_count + 1):
            if (i == user):
                continue
            # 这个条件到时候可以修改
            if self.userSimMatrix[user][i] > 0:
                Nu.add(i)

        Nuj = Nu & self.item_users[item]
        sum_rating = 0.0
        sum_swu = 0.0
        # 如果Nuj为空，应该怎么办???
        if (len(Nuj) == 0):
            return self.user_mean[user]

        if (len(Nuj) > k):
            Nuj = sorted(Nuj, key=lambda i: self.userSimMatrix[user][i], reverse=True)[:k]
        for i in Nuj:
            sum_rating += (self.trainData[i][item] - self.user_mean[i]) * self.userSimMatrix[user][i]
            sum_swu += abs(self.userSimMatrix[user][i])
        sum_rating = sum_rating / sum_swu + self.user_mean[user]
        return sum_rating

    # 预测所有用户对所有物品的评分
    def predict(self, k=8):
        self.readData()
        self.preprocessData()
        self.userSimilarity()
        # user_count = len(self.user_mean)
        # item_count = len(self.item_users)
        # for i in range(1,user_count+1):
        #     for j in range(1,item_count+1):
        #         if j not in self.trainData[i].keys():
        #             predicted_rating = self.ratingPrediction(i,j,k)
        #             if(predicted_rating>5):
        #                 predicted_rating= 5
        #             if(predicted_rating<1):
        #                 predicted_rating= 1
        #             self.trainData[i][j] = predicted_rating

    # 计算RMSE和MAE
    def test(self, dataFileName):
        self.data = pd.read_csv(dataFileName, sep='\t', header=None,
                                names=['user_id', 'item_id', 'rating', 'timestamp'])
        for index, row in self.data.iterrows():
            self.testData.setdefault(row['user_id'], {})
            self.testData[row['user_id']][row['item_id']] = row['rating']
        mae = 0.0
        rmse = 0.0
        count = 0
        with open('MCF_output/u1_UCF.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for user, item in self.testData.items():
                for i in item.keys():
                    predicted_rating = self.ratingPrediction(user, i, 50)
                    if (predicted_rating > 5):
                        predicted_rating = 5
                    if (predicted_rating < 1):
                        predicted_rating = 1
                    writer.writerow([user, i, predicted_rating])
                    count += 1
                    error = self.testData[user][i] - predicted_rating
                    mae += abs(error)
                    rmse += error ** 2
        mae = mae / count
        rmse = (rmse / count) ** 0.5
        print('MAE:', mae)
        print('RMSE:', rmse)
    # def readPredictedMatrixAndTest(self, matrixFileName,testDataFileName):
    #     matrix = pd.read_csv('MCF_output/UCF_predictedRatingMatrix.csv',engine='python', sep='\t', header=None)
    #     self.data = pd.read_csv(testDataFileName, sep='\t', header=None,
    #                             names=['user_id', 'item_id', 'rating', 'timestamp'])
    #     mae = 0.0
    #     rmse = 0.0
    #     count = 0
    #     print(matrix.head(2))
    #     # for index, row in self.data.iterrows():
    #     #     count += 1
    #     #     error = row['rating'] - matrix[row['user_id']][row['item_id']]
    #     #     mae += abs(error)
    #     #     rmse += error**2
    #     # mae = mae/count
    #     # rmse = (rmse/count)**0.5
    #     # print('MAE:',mae)
    #     # print('RMSE:',rmse)


if __name__ == '__main__':
    print('UCF:')
    ucf = UserBasedCF('input/ml-100k/u1.base')
    # ucf.readPredictedMatrixAndTest('MCF_output/UCF_predictedRatingMatrix.csv','input/ml-100k/u1.test')
    start_time = time.time()
    ucf.predict(50)
    ucf.test('input/ml-100k/u1.test')
    end_time = time.time()
    print('cost %f seconds' % (end_time - start_time))
