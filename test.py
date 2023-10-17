from utils import read_to_one_class, grouping
from typing import List, Tuple, Dict, Set
import heapq


class OCCF:
    train_d: List[Tuple[int, int]]
    users: Dict[int, Set[int]]
    items: Dict[int, Set[int]]
    K: int

    def __init__(self, train: List[Tuple[int, int]], K: int):
        self.train_d = train
        self.users = grouping(self.train_d, 'user')
        self.items = grouping(self.train_d, 'item')
        self.K = K

    def predict(self) -> Dict[int, List[int]]:
        pred: Dict[int, List[Tuple[int, float]]] = {}
        for u, item_set in self.items.items():
            pred_u = []
            for j, user_set in self.users.items():
                if j not in item_set:
                    pred_rate = len(user_set)  # / n - mu
                    pred_u.append((j, pred_rate))
            pred_u = heapq.nlargest(self.K, pred_u, key=lambda x: x[1])
            pred_u = [j for j, _ in pred_u]
            pred[u] = pred_u
        return pred

    def evaluate(self, test_d: List[Tuple[int, int]],
                 pred: Dict[int, List[int]]) -> Dict[str, float]:
        real = grouping(test_d, 'user')
        test_len = len(test_d)
        user_len = len(real)
        # Pre@K
        pre, cnt = 0.0, 0
        for u, item_set in real.items():
            if u in pred:
                pre += len(item_set.intersection(pred[u])) / self.K
                cnt += 1
        pre /= cnt
        return {
            'Pre@%d' % self.K: pre
        }


if __name__ == '__main__':
    train_d = read_to_one_class('ml-100k/u1.base', lambda x: x >= 4)
    test_d = read_to_one_class('ml-100k/u1.test', lambda x: x >= 4)
    occf = OCCF(train_d, 5)
    pred = occf.predict()
    eval = occf.evaluate(test_d, pred)
    print(eval)