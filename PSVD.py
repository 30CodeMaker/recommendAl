import numpy as np
import pandas as pd
def svd(A, k):
    U, s, V = np.linalg.svd(A)
    Uk = U[:, :k]
    sk = np.diag(s[:k])
    Vk = V[:k, :]
    return Uk @ sk @ Vk

# 示例
u1_base = pd.read_csv('input/ml-100k/u1.base', sep='\t', header=None
                      ,names=['user_id', 'item_id', 'rating', 'timestamp'])
u1_test = pd.read_csv('input/ml-100k/u1.test', sep='\t', header=None
                      ,names=['user_id', 'item_id', 'rating', 'timestamp'])

u_mean = u1_base.groupby('user_id')['rating'].mean()
userItem = np.zeros((944, 1683))
for index,row in u1_base.iterrows():
    userItem[row['user_id']][ row['item_id']] = row['rating'] - u_mean[row['user_id']]

k = 20
R_ = svd(userItem, k)
mae = 0.0
rsme = 0.0
for index,row in u1_test.iterrows():
    r_prediction = R_[row['user_id'], row['item_id']] + u_mean[row['user_id']]
    r_prediction = min(5, r_prediction)
    r_prediction = max(1, r_prediction)
    mae += abs(r_prediction - row['rating'])
    rsme += (r_prediction - row['rating']) ** 2
mae /= len(u1_test)
rsme /= len(u1_test)
rsme = np.sqrt(rsme)
print(f"PSVD(k={k}):")
print('MAE =', mae)
print('RSME =', rsme)