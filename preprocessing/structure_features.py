
import networkx as nx
import dgl
from typing import Union
import numpy as np
import pandas as pd
import multiprocessing as mp
from utils import logger 

"""
compute the structure features of node

"""
def multi_processing(fun_list, graph, node_idx=None):
    """

    """
    jobs = []
    pool = mp.Pool(len(fun_list)) if len(fun_list) < mp.cpu_count() else mp.Pool(int(mp.cpu_count() / 2))
    for fun in fun_list:
        logger.debug(f"start centrality/clustering function {fun.__name__}")

        p = pool.apply_async(fun, args=(graph,))

        jobs.append(p)
    
    pool.close() 
    pool.join()
    if not node_idx:
        result =  [job.get(timeout=3600000) for job in jobs]
    else:
        result = [job.get(timeout=3600000)[node_idx] for job in jobs]
    
    return result


def graph_features_extracting(graph: Union[nx.DiGraph, nx.Graph], 
                              use_mp: Union[bool, int]=False):
    """
    implemented for directed type is allowed 
    """
    logger.debug("call graph_features_extracting...")

    fun_list = [nx.density, 
                # nx.average_clustering, 
                nx.number_of_selfloops, 
                nx.transitivity]

    if use_mp:
        result = multi_processing(fun_list, graph)
    else:
        result = [fun(graph) for fun in fun_list]

    logger.debug("graph_features_extracting ended")
    return result


def node_features_extracting(graph: Union[nx.DiGraph, nx.Graph], 
                             node_idx, 
                             use_mp: Union[bool,int] = False):
    logger.debug("call node_features_extracting...")
    cen_list = []
    if isinstance(graph, nx.DiGraph): 
        # cen_list += [nx.in_degree_centrality, nx.out_degree_centrality]
        graph = graph.to_undirected() 

    # cen_list += [nx.degree_centrality]
    
    cen_list += [
        # nx.eigenvector_centrality,  # O(|V|^3), discard for its high computation complexity
        # nx.closeness_centrality,   # O(|V|^2*log|V|+|V|*|E|) with Johnson or Brandes algorithm, O(|V|^3) at worst
        # nx.betweenness_centrality  # O(|V|*|E|)
                ]

    clu_list = [nx.triangles, nx.clustering, nx.square_clustering]
    fun_list = cen_list + clu_list

    if use_mp:
        result = multi_processing(fun_list,graph, node_idx=node_idx)
    else:
        result = [fun(graph)[node_idx] for fun in fun_list]

    logger.debug("node_features_extracting ended.")
    return result


# def neighbour_averaging(origin_feat: Union[torch.Tensor, np.ndarray], 
#                         neighbours_set: Union[list,tuple,set], 
#                         aggregating: str="mean",
#                         to_tensor: Union[int, bool] = True) -> torch.Tensor:
#     logger.debug("call neighbour_averaging...")
#     if isinstance(neighbours_set, set):
#         neighbours_set = list(neighbours_set)
#     if isinstance(origin_feat, np.ndarray):

#         if aggregating == "mean":
#             features = origin_feat[neighbours_set].mean(axis=0)
#         elif aggregating == "max":
#             features = origin_feat[neighbours_set].max(axis=0)
#         elif aggregating == "sum":
#             features = origin_feat[neighbours_set].sum(axis=0)
#         else:
#             raise KeyError
#         features = torch.Tensor(features) 

#     elif isinstance(origin_feat, torch.Tensor):
#             if aggregating == "sum":
#                 features = torch.sum(origin_feat[neighbours_set], dim=0)
#             elif aggregating == "mean":
#                 features = torch.mean(origin_feat[neighbours_set],dim=0)
#             elif aggregating == "max":
#                 features = torch.max(origin_feat[neighbours_set],dim=0).values
#             else:
#                 raise KeyError
#     else:
#         raise TypeError
#     if to_tensor and not isinstance(features, torch.Tensor):
#         features = torch.Tensor(features) 
#     else:
#         features = features.tolist()
#     logger.debug("neighbour_averaging ended.")
#     return features  


def get_inception_structure_features(graph: Union[nx.DiGraph, nx.Graph],
                #    origin_feat: torch.Tensor,
                   n_hops: int = 3,
                   node_idx: int = None
                   ):
             
    logger.debug("call eog_graph_features_extracting...")

    # * will convert graph to nx.MultiGraph, ignored 
    # if isinstance(graph, dgl.DGLGraph):
    #     graph = graph.to_networkx()
    if isinstance(graph, nx.DiGraph):
        undirected = False
    else:
        undirected = True 
        
    graph_features_list = []
    node_features_list = [] 
    # previous_neighbours_set = {node_idx}
    for i in range(n_hops):
        
        egonet = nx.ego_graph(graph, node_idx, radius=i+1, center=True, undirected=undirected, distance=None)
        # 1, compute graph features 

        graph_features = graph_features_extracting(egonet)
        graph_features_list += graph_features
        # 2, compute node features
        node_features = node_features_extracting(egonet, node_idx)
        node_features_list += node_features
        # # 3, compute neighbour averaging features 
        # if averaging_feature:
        #     neighbours_set = set(egonet.nodes()) - previous_neighbours_set
        #     neighbour_features = neighbour_averaging(origin_feat=origin_feat, neighbours_set=neighbours_set, aggregating=aggregating)
        #     previous_neighbours_set = set(egonet.nodes())
        #     features_list['average'] = neighbour_features
    return graph_features_list, node_features_list 
    # if features_list:

    #     logger.debug("ego_graph_features_extracting ended.")
    #     return features_list
    # else:
    #     logger.error("the combination of parameters make the generated features empty!")
    #     raise KeyError


