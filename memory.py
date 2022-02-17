#### Code modified from https://github.com/abderraouf2che/RL-Traffic-Signal-Control/blob/main/Memory.py  #####

import random
import numpy as np

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, size_min):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self._size_min = size_min

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add_sample(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def get_samples(self, n):
        # if self._size_now() < self._size_min:
        #     return []

        # if n > self._size_now():
        #     return random.sample(self._samples, self._size_now())  # get all the samples
        # else:   
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
    
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
    
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
    
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
    
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
    
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
    # def _size_now(self):
    #     """
    #     Check how full the memory is
    #     """
    #     print("tree:", np.array(self.tree).shape))
    #     return int(np.array(self.tree).shape)

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])








# import heapq
# import torch
# from itertools import count
# from collections import deque
# tiebreaker = count()

# class Memory:
#     def __init__(self, size_max, size_min):
#         self._samples = []
#         self._size_max = size_max
#         self._size_min = size_min


#     def add_sample(self, TD, transition):
#         """
#         Add a sample into the memory
#         """
#         # self._samples.append(transition)
#         heapq.heappush(self._samples, (-TD, next(tiebreaker), transition))
#         heapq.heapify(self._samples)
#         if self._size_now() > self._size_max:
#             self._samples.pop(0)  # if the length is greater than the size of memory, remove the oldest element


#     def get_samples(self, n, model,train=0):
#         """
#         Get n samples randomly from the memory
#         """
        # if self._size_now() < self._size_min:
        #     return []

        # if n > self._size_now():
        #     return random.sample(self._samples, self._size_now())  # get all the samples
        # else:
#             x = random.sample(self._samples, 10*n)  # get "batch size" number of samples
#             # if self._size_now() > self._size_max:
#             #     self._samples = self._samples[:-1]
#             batch = heapq.nsmallest(n, x)
#             batch = [e for (_, _, e) in batch]
#             # print(batch)

#             del self._samples[0:n]
#             # self._samples = self._samples[n:]
            
#             return batch


    # def _size_now(self):
    #     """
    #     Check how full the memory is
    #     """
    #     return len(self._samples)
