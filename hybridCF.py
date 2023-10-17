import pandas as pd
import time

if __name__ == '__main__':
    print('HybridCF:')
    Lamda = 0.5
    start_time = time.time()
    UCF = pd.read_csv('MCF_output/u1_UCF.csv', engine='python', sep=',', header=None,
                      names=['user_id', 'item_id', 'rating'])
    ICF = pd.read_csv('MCF_output/u1_ICF.csv', engine='python', sep=',', header=None,
                      names=['user_id', 'item_id', 'rating'])
    hybrid_matrix = pd.merge(UCF, ICF, on=['user_id', 'item_id'])
    hybrid_matrix['rating'] = Lamda * hybrid_matrix['rating_x'] + (1 - Lamda) * hybrid_matrix['rating_y']
    hybrid_matrix = hybrid_matrix[['user_id', 'item_id', 'rating']]
    test_data = pd.read_csv('input/ml-100k/u1.test', engine='python', sep='\t', header=None,
                            names=['user_id', 'item_id', 'rating', 'timestamp'])
    hybrid_matrix = pd.merge(hybrid_matrix, test_data, on=['user_id', 'item_id'])

    mae = 0.0
    rmse = 0.0
    count = 0
    for index, row in hybrid_matrix.iterrows():
        count += 1
        error = row['rating_x'] - row['rating_y']
        mae += abs(error)
        rmse += error**2
    mae = mae/count
    rmse = (rmse/count)**0.5
    print('MAE:',mae)
    print('RMSE:',rmse)
