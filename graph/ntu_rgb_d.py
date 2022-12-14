import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 50
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12), (4, 29),

                    (26, 27), (27, 46), (28, 46), (29, 28), (30, 46), (31, 30), (32, 31),
                    (33, 32), (34, 46), (35, 34), (36, 35), (37, 36), (38, 26),
                    (39, 38), (40, 39), (41, 40), (42, 26), (43, 42), (44, 43),
                    (45, 44), (47, 48), (48, 33), (49, 50), (50, 37)
                    ]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
