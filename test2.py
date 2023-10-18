import heapq

def top_k_heap(nums, k):
    """
    :param nums: List[int]
    :param k: int
    :return: List[int]
    """
    heap = []
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        else:
            if num > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, num)
    return heap

if __name__ == '__main__':
    nums = [3, 5, 1, 2, 4, 6,11,22,33,10,9,8,7]
    k = 3
    print(top_k_heap(nums, k))
