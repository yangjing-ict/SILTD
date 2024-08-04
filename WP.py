import argparse
import os
from sklearn import metrics
import numpy as np
from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning, EncodingTree
from torch_scatter import scatter_sum
from util import multirank_cuda
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import time

def process(filepath, LLM):
    path = os.path.join(filepath, f"metric_features_{LLM}.pkl")
    print(path)
    label_path = os.path.join(filepath, f"labels_{LLM}.pkl")
    features, labels = [], []
    if os.path.exists(path):
        feature_sets = np.load(path, allow_pickle=True)
    if os.path.exists(label_path):
        labels = np.load(label_path, allow_pickle=True)
    num = len(labels)
    
    data = {}
    features = ["LL", "LR", "LRR", "DetectGPT", "NPR"]
    for idx, set in enumerate(feature_sets):
        data[features[idx]] = set

    data['labels'] = labels
    
    return data, num

def get_threshold(feature, LLM):
    thresholds = {
        "LL": {"base": -2.7, "Claude": -3.1, "Dolly": -2.9, "StableLM": -2.9},
        "LRK": {"base": -1.4, "Claude": -1.5, "StableLM": -1.5},
        "LRR": {"base": -2.01, "ChatGLM": -2.1, "Claude": -1.9, "StableLM": -1.98},
        "DG": {"base": 0.9, "Claude": 0.3, "Dolly": 0.3, "StableLM": 0.1},
        "NPR": {"base": 1, "ChatGPT": 1.03, "StableLM": 1.01, "Dolly": 1.02, "ChatGLM": 1.1}
    }
    return thresholds[feature].get(LLM, thresholds[feature]["base"])

def test(filepath, LLM, feature1, feature2=None, feature3=None):
    data, num = process(filepath, LLM)

    LL = data["LL"]
    LRR = data["LRR"]
    LR = data["LR"]
    DetectGPT = data["DetectGPT"]
    NPR = data["NPR"]
    labels = data['labels']

    # 构图
    graph = np.zeros((2, num, num))
    global adj_matrix
    adj_matrix = np.zeros((num, num))

    def construct_graph(feature, graph_idx):
        global adj_matrix
        if feature == "LL":
            diff_LL = abs(LL.reshape(-1, 1) - LL.reshape(1, -1))
            max_LL = np.maximum(LL.reshape(-1, 1), LL.reshape(1, -1))
            min_LL = np.minimum(LL.reshape(-1, 1), LL.reshape(1, -1))
            print(diff_LL)
            mask_LL = (diff_LL < 0.2 * abs(max_LL)) & (min_LL > get_threshold(feature, LLM))
            graph[graph_idx][mask_LL] = 1
            graph[graph_idx] -= np.diag(np.diag(graph[graph_idx]))
    
            adj_matrix += (1 - diff_LL / max_LL) * mask_LL

        elif feature == "LRK":
            diff_LR = abs(LR.reshape(-1, 1) - LR.reshape(1, -1))
            max_LR = np.maximum(LR.reshape(-1, 1), LR.reshape(1, -1))
            min_LR = np.minimum(LR.reshape(-1, 1), LR.reshape(1, -1))
            print(diff_LR)
            mask_LR = (diff_LR < 0.001 * abs(max_LR)) & (min_LR > get_threshold(feature, LLM))
            graph[graph_idx][mask_LR] = 1
            graph[graph_idx] -= np.diag(np.diag(graph[graph_idx]))

            adj_matrix += (1 - diff_LR / max_LR) * mask_LR

        elif feature == "LRR":
            diff_LRR = abs(LRR.reshape(-1, 1) - LRR.reshape(1, -1))
            max_LRR = np.maximum(LRR.reshape(-1, 1), LRR.reshape(1, -1))
            min_LRR = np.minimum(LRR.reshape(-1, 1), LRR.reshape(1, -1))
            print(diff_LRR)
            mask_LRR = (diff_LRR < 0.1 * abs(max_LRR)) & (max_LRR < get_threshold(feature, LLM))
            graph[graph_idx][mask_LRR] = 1
            graph[graph_idx] -= np.diag(np.diag(graph[graph_idx]))

            adj_matrix += (1 - diff_LRR / max_LRR) * mask_LRR

        elif feature == "DG":
            diff_DetectGPT = abs(DetectGPT.reshape(-1, 1) - DetectGPT.reshape(1, -1))
            max_DetectGPT = np.maximum(DetectGPT.reshape(-1, 1), DetectGPT.reshape(1, -1))
            min_DetectGPT = np.minimum(DetectGPT.reshape(-1, 1), DetectGPT.reshape(1, -1))
            print(diff_DetectGPT)
            mask_DetectGPT = (diff_DetectGPT < 0.4) & (min_DetectGPT > get_threshold(feature, LLM))
            graph[graph_idx][mask_DetectGPT] = 1
            graph[graph_idx] -= np.diag(np.diag(graph[graph_idx]))

            adj_matrix += (1 - diff_DetectGPT) * mask_DetectGPT

        elif feature == "NPR":
            diff_NPR = abs(NPR.reshape(-1, 1) - NPR.reshape(1, -1))
            max_NPR = np.maximum(NPR.reshape(-1, 1), NPR.reshape(1, -1))
            min_NPR = np.minimum(NPR.reshape(-1, 1), NPR.reshape(1, -1))
            print(diff_NPR)
            mask_NPR = (diff_NPR < 0.1 * abs(max_NPR)) & (min_NPR > get_threshold(feature, LLM))
            graph[graph_idx][mask_NPR] = 1
            graph[graph_idx] -= np.diag(np.diag(graph[graph_idx]))

            adj_matrix += (1 - diff_NPR / max_NPR) * mask_NPR

    if feature1:
        construct_graph(feature1, 0)
    if feature2:
        construct_graph(feature2, 1)

    adj_matrix -= np.diag(np.diag(adj_matrix))
    G = nx.Graph(adj_matrix)

    index = np.where(np.sum(adj_matrix, axis=1) == 0) # 和为 0 的行索引
    for i in index:
        adj_matrix[i, i] = 0.01

    # 计算mulrirank
    if feature1 and feature2: x, y = multirank_cuda(graph) 
    else: x, y = multirank_cuda(graph[[0]])

    # 编码树
    edges = np.array(adj_matrix.nonzero())  # [2, E]
    ew = adj_matrix[edges[0, :], edges[1, :]]
    ew, edges = torch.Tensor(ew), torch.tensor(edges).t()
    dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])  # dist/2=di
    dist = dist / (2 * ew.sum())  # ew.sum()=vol(G) dist=di/vol(G)

    g = GraphSparse(edges, ew, dist)
    optim = OperatorPropagation(Partitioning(g, None))
    # optim.perform(p=0.15)
    optim.perform()
    division = optim.enc.node_id
    SE2d = optim.enc.structural_entropy(reduction='sum', norm=True)
    module_se = optim.enc.structural_entropy(reduction='module', norm=True)
    total_cluster = torch.max(division) + 1
    clusters = {}
    for i in range(total_cluster):
        idx = division == i
        if idx.any():
            clusters[i] = idx.nonzero().squeeze(1)

    LLM_rate = []
    LLM_list=[]
    pre_LLM = np.zeros(num)
    value = np.zeros(num)

    # 簇划分
    num_LLM_cluster = 0
    node_rep = []
    for i in clusters.keys():
        c = clusters[i]
        n_LLMs = 0
        n_nodes = 0
        n_x = 0
        for node in c:
            n_LLMs += labels[node]
            n_nodes += 1
            n_x += x[node]
        cluster_SE = module_se[i]

        n_beta = (n_x / n_nodes) / (1 / num) * 0.9 + cluster_SE / (sum(module_se) / total_cluster) * 0.1
        if n_beta >= 0.60:
            num_LLM_cluster += 1
            for node in c:
                pre_LLM[node] = 1
                value[node] = n_beta
                node_rep.append([x[node], labels[node], pre_LLM[node]])
        else:
            for node in c:
                LLM_list.append(node)
                value[node] = n_beta
                node_rep.append([x[node], labels[node], pre_LLM[node]])
        LLM_rate.append([n_LLMs / n_nodes, n_LLMs, n_nodes, n_x / n_nodes, cluster_SE, n_beta])

    # 准确率计算
    acc = metrics.accuracy_score(labels, pre_LLM)
    precision = metrics.precision_score(labels, pre_LLM)
    recall = metrics.recall_score(labels, pre_LLM)
    f1 = metrics.f1_score(labels, pre_LLM)
    fpr, tpr, thresholds = metrics.roc_curve(labels, value)
    auc_score = metrics.roc_auc_score(labels, value)
    print('acc:{}'.format(acc))
    print('Precision:{}'.format(precision))
    print('Recall:{}'.format(recall))
    print('F1:{}'.format(f1))
    print('AUC:{}'.format(auc_score))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--LLM', type=str, default="Dolly", help="ChatGPT, ChatGLM, Claude, Dolly, StableLM, GPT4All")
    parser.add_argument('--feature1', type=str, default="LL", help="LL, LRK, LRR, DG, NPR")
    parser.add_argument('--feature2', type=str, default=None, help="LL, LRK, LRR, DG, NPR")

    args = parser.parse_args()

    start = time.time()
    LLM = args.LLM
    feature1 = args.feature1
    feature2 = args.feature2
    feature3 = args.feature3

    test(f'/home/yangjing/AIGCDetect/extract_features/features/WP', LLM, feature1, feature2, feature3)
    print(time.time() - start)
