import csv
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import time

class ItemBasedCF:
    def __init__(self, datafile):
        self.datafile = datafile
        self.data =pd.DataFrame()
        self.trainData = {}
        self.testData = {}
        self.itemSimMatrix = []
        self.user_mean= {}
        # 物品用户倒排表
        self.item_users = dict()
        # 用户物品倒排表
        self.user_items = dict()
        self.item_count = 1682
    #read data from input_file
    def readData(self):
        self.data=pd.read_csv(self.datafile, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
        user_group = self.data.groupby('user_id')
        #meanUserScores
        self.user_mean = user_group['rating'].mean()
        item_group = self.data.groupby('item_id')
        #meanItemScores
        self.item_mean = item_group['rating'].mean()
        traindata_list = {}
        # 存储格式：
        for index, row in self.data.iterrows():
            traindata_list.setdefault(row['user_id'], {})
            traindata_list[row['user_id']][row['item_id']] = row['rating']
        self.trainData = traindata_list

    #计算物品相似度
    def itemsim(self, item1, item2):
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        #求item1和item2评分用户的交集
        self.item_users.setdefault(item1, set())
        self.item_users.setdefault(item2, set())
        item_users_intersection = self.item_users[item1] & self.item_users[item2]
        if len(item_users_intersection) == 0:
            return 0
        #遍历item1和item2评分用户的交集
        for user in item_users_intersection:
            sum_x += (self.trainData[user][item1]-self.user_mean[user])**2
            sum_y += (self.trainData[user][item2]-self.user_mean[user])**2
            sum_xy += (self.trainData[user][item1]-self.user_mean[user])*(self.trainData[user][item2]-self.user_mean[user])
            # sum_x+=self.trainData[user][item1]**2
            # sum_y+=self.trainData[user][item2]**2
            # sum_xy+=self.trainData[user][item1]*self.trainData[user][item2]

        # ???? 如果分母为0，怎么办
        if(sum_x==0 or sum_y==0):
            return 0
        return sum_xy/((sum_x*sum_y)**0.5)

    #计算物品相似度矩阵
    def itemSimilarity(self):
        item_count = self.item_count

        #求物品用户倒排表和用户物品倒排表
        for u, item in self.trainData.items():
            self.user_items.setdefault(u,set())
            for i in item.keys():
                self.user_items[u].add(i)
                self.item_users.setdefault(i,set())
                self.item_users[i].add(u)

        #计算物品相似度矩阵（皮尔逊相关系数）
        self.itemSimMatrix = np.zeros((item_count+1,item_count+1))
        for i in range(1,item_count+1):
            for j in range(i,item_count+1):
                if i == j:
                    self.itemSimMatrix[i][j] = 1
                else:
                    self.itemSimMatrix[i][j] = self.itemSimMatrix[j][i] = self.itemsim(i,j)

    #预测用户对物品的评分
    def ratingPrediction(self,uesr,item,k=8):
        Nj = set()
        self.user_items.setdefault(uesr, set())
        item_count = self.item_count
        for i in range(1,item_count+1):
            if i==item:
                continue
            # 这个条件到时候可以修改
            if self.itemSimMatrix[item][i] > 0:
                Nj.add(i)

        Nuj = Nj & self.user_items[uesr]
        if(len(Nuj)==0):
            return self.user_mean[uesr]
        if(len(Nuj)>k):
            Nuj = set(sorted(Nuj, key=lambda x:self.itemSimMatrix[item][x], reverse=True)[0:k])

        sum_x = 0.0
        sum_y = 0.0
        for j in Nuj:
            sum_x += self.itemSimMatrix[item][j]*(self.trainData[uesr][j])
            sum_y += self.itemSimMatrix[item][j]
        #????? 如果分母为0，怎么办
        if sum_y == 0:
            return self.user_mean[uesr]
        return sum_x/sum_y

    #预测所有用户对所有物品的评分
    def predict(self,k=8):
        self.readData()
        self.itemSimilarity()
        # user_count = len(self.user_mean)
        # item_count = len(self.item_mean)
        # for user in range(1,user_count+1):
        #     for item in range(1,item_count+1):
        #         if item in self.trainData[user].keys():
        #             continue
        #         predicted_rating = self.ratingPrediction(user,item,k)
        #         if (predicted_rating > 5):
        #             predicted_rating = 5
        #         if (predicted_rating < 1):
        #             predicted_rating = 1
        #         self.trainData[user][item] = predicted_rating
        # df = pd.DataFrame.from_dict(self.trainData, orient='index')
        # df.to_csv('MCF_output/ICF_predictedRatingMatrix.csv')

    #计算RMSE和MAE
    def test(self,testDataFileName):
        self.data = pd.read_csv(testDataFileName, sep='\t', header=None,
                                names=['user_id', 'item_id', 'rating', 'timestamp'])
        for index, row in self.data.iterrows():
            self.testData.setdefault(row['user_id'], {})
            self.testData[row['user_id']][row['item_id']] = row['rating']
        mae = 0.0
        rmse = 0.0
        count = 0
        with open('MCF_output/u1_ICF.csv', 'w', newline='') as csvfile:
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
        mae = mae/count
        rmse = (rmse/count)**0.5
        print('MAE:',mae)
        print('RMSE:',rmse)

if __name__ == '__main__':
    print('ICF:')
    icf = ItemBasedCF(('input/ml-100k/u1.base'))
    start_time = time.time()
    icf.predict(50)
    icf.test('input/ml-100k/u1.test')
    end_time = time.time()
    print('cost %f seconds' % (end_time - start_time))