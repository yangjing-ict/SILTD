import ijson
import numpy as np
import csv
import os
import torch

def multirank_cuda(multi_graph):
    """
    迭代得到x_new, y_new
    x_new: 节点的影响力
    y_new: 每个异构图的权重
    """
    # multi_graph: [r, m, m]
    # print('----------------------------------------------')
    print('is calculating multirank...')
    devices = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    multi_graph = torch.tensor(multi_graph, device=devices)
    m = len(multi_graph[0])
    r = len(multi_graph)

    sum_i1 = torch.sum(multi_graph, dim=1)
    sum_j = torch.sum(multi_graph, dim=0)
    O = multi_graph / sum_i1.reshape(r, 1, m)  # m, m, r
    R = multi_graph / sum_j.reshape(1, m, m)  # r, m, m
    O[torch.tile(sum_i1.reshape(r, 1, m) == 0, (1, m, 1))] = 1 / m
    O = O.permute(1, 2, 0)
    R[torch.tile(sum_j.reshape(1, m, m) == 0, (r, 1, 1))] = 1 / r

    x_old = torch.ones(m, device=devices, dtype=torch.float64)
    x_new = torch.ones(m, device=devices, dtype=torch.float64)
    x_old = x_old * (1 / m)
    y_old = torch.ones(r, device=devices, dtype=torch.float64)
    y_new = torch.ones(r, device=devices, dtype=torch.float64)
    y_old = y_old * (1 / r)
    x_new[:] = x_old[:]
    y_new[:] = y_old[:]
    while True:
        x_old[:] = x_new[:]
        y_old[:] = y_new[:]
        for i in range(m):
            x_new[i] = x_old @ O[i] @ y_old
        for j in range(r):
            y_new[j] = x_old @ R[j] @ x_old
        z = torch.sqrt(sum((x_old - x_new) ** 2)) + torch.sqrt(sum((y_old - y_new) ** 2))
        if z < 0.004:
            break
    print('y_final:', y_new)
    return x_new.cpu().detach().numpy(), y_new.cpu().detach().numpy()

