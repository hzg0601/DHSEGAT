'''
Description: 
version: 
Author: huangzg
LastEditors: huangzg
Date: 2021-09-18 14:12:32
LastEditTime: 2021-09-26 15:19:58
'''
from typing import Union 
import numpy as np
import networkx as nx
import dgl 
import scipy.stats as sci 



def distribution_feats(weight_list):

    if len(weight_list) == 0:
        dist_feat = [0]*7
    elif len(weight_list) == 1:
        dist_feat = weight_list + [0]*6
    elif len(weight_list) == 2:
        dist_feat = [np.max(weight_list), 0,
                    np.mean(weight_list), np.min(weight_list),
                    np.std(weight_list), sci.skew(weight_list), sci.kurtosis(weight_list)]

    else:
        dist_feat = [np.max(weight_list), np.median(weight_list),
                     np.mean(weight_list), np.min(weight_list),
                     np.std(weight_list), sci.skew(weight_list), sci.kurtosis(weight_list)]
    return dist_feat


def sp_feature_extracting(gen_graph, node_idx):
    shortest_path = nx.shortest_path_length(gen_graph,source=node_idx)
    shortest_path[node_idx] = 0
    shortest_path = [length for node_idx, length in shortest_path.items()]
    return distribution_feats(shortest_path)    


def merge_neigh(distance_feature: np.ndarray) -> np.ndarray:
    distance_mean = distance_feature.mean(axis=0)
    distance_sum = distance_feature.sum(axis=0)
    distance_max = distance_feature.max(axis=0)
    
    return np.stack([distance_mean, distance_sum, distance_max],axis=0)


def rw_feature_extracting(adj: np.ndarray, node_set: Union[int, list], rw_depth: int ):
    node_set = [node_set] if not isinstance(node_set, list) else node_set
    epsilon = 1e-6
    if isinstance(adj, nx.DiGraph) or isinstance(adj, nx.Graph):
        node_ids_dict = {node_id:id for id,node_id in enumerate(adj.nodes())}
        adj = nx.adjacency_matrix(adj).toarray()
        node_set = [node_ids_dict[node] for node in node_set] 
        
    adj = adj / (adj.sum(1, keepdims=True) + epsilon)

    rw_list = [np.identity(adj.shape[0])[node_set]]
    for _ in range(rw_depth):
        rw = np.matmul(rw_list[-1], adj) 
        rw_list.append(rw) 
    features_rw_tmp = np.stack(rw_list, axis=2)  # shape len(set_size) * adj.shape[0] * rw_depth
    # pooling
    features_rw = features_rw_tmp.sum(axis=0)  # shape adj.shape[0] * rw_depth 
    features_rw = merge_neigh(features_rw) 
    return features_rw   


def get_distance_features(graph, n_hops, node_idx=None, rw_depth=3):

    if isinstance(graph, nx.DiGraph):
        undirected = False
    else:
        undirected = True 
    
    egonet = nx.ego_graph(graph, node_idx, radius=n_hops, center=True, undirected=undirected, distance=None)
    
    sp_feat = sp_feature_extracting(egonet, node_idx)
    # rw_feat = rw_feature_extracting(egonet, node_idx, rw_depth)
    
    return sp_feat

# def sp_feature_extracting(G:Union[nx.Graph, nx.DiGraph], node_set: Union[int, list], max_sp:int):
#     node_set = [node_set] if not isinstance(node_set, list) else node_set
#     dim = max_sp + 2 
#     set_size = len(node_set)  
#     sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1  
#     for i, node in enumerate(node_set):
       
#         for node_ngh, (node_ngh_ori, length) in enumerate(nx.shortest_path_length(G, source=node).items()):
#             sp_length[node_ngh, i] = length  
#     sp_length = np.minimum(sp_length, max_sp)  # [node_nghs , len(node_set)]
   
#     onehot_encoding = np.eye(dim, dtype=np.float64)  # [dim, dim]
#     features_sp = onehot_encoding[sp_length].sum(axis=1)  # [node_nghs,len(node_set), dim] -> [node_nghs, dim]
#     features_sp = merge_neigh(features_sp) # [3,dim]
#     return features_sp


# def rw_feature_extracting(adj: np.ndarray, node_set: Union[int, list], rw_depth: int ):
#     node_set = [node_set] if not isinstance(node_set, list) else node_set
#     epsilon = 1e-6
#     if isinstance(adj, nx.DiGraph) or isinstance(adj, nx.Graph):
#         node_ids_dict = {node_id:id for id,node_id in enumerate(adj.nodes())}
#         adj = nx.adjacency_matrix(adj).toarray()
#         node_set = [node_ids_dict[node] for node in node_set] 
        
#     adj = adj / (adj.sum(1, keepdims=True) + epsilon)

#     rw_list = [np.identity(adj.shape[0])[node_set]]
#     for _ in range(rw_depth):
#         rw = np.matmul(rw_list[-1], adj) 
#         rw_list.append(rw) 
#     features_rw_tmp = np.stack(rw_list, axis=2)  # shape len(set_size) * adj.shape[0] * rw_depth
#     # pooling
#     features_rw = features_rw_tmp.sum(axis=0)  # shape adj.shape[0] * rw_depth 
#     features_rw = merge_neigh(features_rw) 
#     return features_rw    

# def ego2sentence(gen_graph: Union[nx.Graph, nx.DiGraph], node_id, feats=None, max_n_nodes=1024):
#     # similar with position embedding 
    

#     shortest_path = nx.shortest_path_length(gen_graph,source=node_id)
#     shortest_path[node_id] = 0
#     shortest_path = np.array([(node_idx, length) for node_idx, length in shortest_path.items()])
#     # sort by shortest_path_distance
#     shortest_path = shortest_path[shortest_path[:,1].argsort()]
#     if len(shortest_path) > max_n_nodes:
#         shortest_path = shortest_path[:max_n_nodes,:]
#         gen_graph.remove_nodes_from(shortest_path[max_n_nodes:, 0])
#     if feats is not None:
#         nodes_feat = feats[shortest_path[:,0]]
        
#         return shortest_path, gen_graph, nodes_feat
#     else:
#         return shortest_path, gen_graph
