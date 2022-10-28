"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import numpy as np


num_node = 33
self_link = [(i, i) for i in range(num_node)]
in_edge = [(0,1), (1,2), (2, 3), (3, 7), (0,4), (4, 5), (5, 6), (6, 8), (0, 10),
           (0, 9), (0, 11), (0, 12), (11, 13), (13, 15), (15, 17), (15, 21),
           (15, 19), (17, 19), (11, 12), (12, 14), (14, 16), (16, 18), (16, 22),
           (16, 20), (18, 20), (11, 23), (23, 25), (23, 24), (25, 27), (27, 29), 
           (27, 31), (29, 21), (12, 24), (24, 26), (26, 28), (28, 30), (28, 32),
           (30, 32)]
out_edge = [(j, i) for (i, j) in in_edge]
neighbor = in_edge + out_edge


def get_hop(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, in_edge, out_edge):
    I = get_hop(self_link, num_node)
    In = normalize_digraph(get_hop(in_edge, num_node))
    Out = normalize_digraph(get_hop(out_edge, num_node))
    A = np.stack((I, In, Out))
    return A


class MediapipeGraph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.in_edge = in_edge
        self.out_edge = out_edge
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, in_edge, out_edge)
        else:
            raise ValueError()
        return A
