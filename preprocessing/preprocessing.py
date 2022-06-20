'''
Description: 
version: 
Author: huangzg
LastEditors: huangzg
Date: 2021-08-21 22:23:35
LastEditTime: 2021-10-19 15:39:09
'''
import multiprocessing as mp
import os
from typing import Union
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import networkx as nx
import numpy as np
from utils import logger 
import argparse
from distance_features import get_distance_features 
from structure_features import get_inception_structure_features

parser = argparse.ArgumentParser(description="preprocessing")

parser.add_argument('--dataset', type=str, default='ogbn-arxiv', choices=['ogbn-arxiv', 'ogbn-products'])
parser.add_argument("--data_dir", type=str, default='../dataset')

# preprocessing
parser.add_argument("--use_mp_pre", type=int, default=0, help="use multiprocess in preprocessing or not")
parser.add_argument("--n_pools", type=int, default=16, help="how many pool in multiprocessing.")
parser.add_argument("--n_hops", type=int, default=3, help="how many hops in generating features")
parser.add_argument('--feat_fun_flag', default='distance', type=str)
parser.add_argument('--to_bidirected', type=int, default=1,help='to_bidirected or not')
# 
args = parser.parse_args() 
logger.info(args)

# -----------------convert dgl.DGLGraph to nx.Graph-----------------------------------------------
dataset = DglNodePropPredDataset(name=args.dataset, root=args.data_dir)
data_dir = os.path.join(args.data_dir, args.dataset.replace("-", "_"), '/processed')
g, labels = dataset[0]

if not args.to_bidirected:
    graph = nx.from_scipy_sparse_matrix(g.adjacency_matrix(scipy_fmt='csr'),create_using=nx.DiGraph)
else:
    graph = nx.from_scipy_sparse_matrix(g.adjacency_matrix(scipy_fmt='csr'),create_using=nx.Graph) 
  
source_node_ids = list(graph.nodes())  

del g, dataset, labels 

# -----------------single processing -----------------------
def batch_dataloader(graph: Union[nx.DiGraph, nx.Graph],
                   node_ids: list = None,
                   n_hops: int = 3,
                   feat_fun_flag="graph"
                   ):

    if feat_fun_flag == "structure":
        feat_fun = get_inception_structure_features
    elif feat_fun_flag == "distance":
        feat_fun = get_distance_features 
    else:
        logger.error("wrong feat fun flag!")
        raise KeyError
    result = []
    for node_id in node_ids:
        temp = feat_fun(graph, 
                        n_hops=n_hops, 
                        node_idx=node_id

                        )
        logger.info(f"preprocessing node {node_id} done.")
        result.append(temp)
    result = np.array(result) 
    return result   
# ---------------------n_batch_feature_multi_sampling------------------------

def sub_feat_generating( node_idx,
                        n_hops,
                        feat_fun
                        ):
    global graph  

    result = feat_fun( graph,
                        n_hops=n_hops,
                        node_idx=node_idx

                        )
    logger.info(f"preprocessing node {node_idx} done.")
    return result 


def multi_feat_generating(source_node_ids,
                        n_hops=args.n_hops,
                        n_pools=args.n_pools,
                        feat_fun_flag=args.feat_fun_flag
                         ):
    # todo call Inspector to generate kwargs
    pools = mp.Pool(n_pools)
    result = []
    if feat_fun_flag == "structure":
        feat_fun = get_inception_structure_features
    elif feat_fun_flag == "distance":
        feat_fun = get_distance_features 
    else:
        logger.error("wrong feat fun flag!")
        raise KeyError
    for node_id in source_node_ids:
        logger.debug(f"start to process node {node_id}")
        p = pools.apply_async(sub_feat_generating, args=(node_id, 
                                                        n_hops,
                                                        feat_fun
                                                        )
                              )
    
        result.append(p)
    result = [p.get(timeout=3600000) for p in result]
    pools.close()
    pools.join()
    result = np.array(result)
    return result 


if __name__ == '__main__':
    
    logger.info(f"start to implement --preprocessing--, the pid is {os.getpid()}")
    if args.use_mp_pre:
        
        feats = multi_feat_generating(source_node_ids,
                                    n_hops=args.n_hops,
                                    n_pools=args.n_pools,
                                    feat_fun_flag=args.feat_fun_flag
                                    )
    else:
        feats = batch_dataloader(graph,
                                 node_ids=source_node_ids,
                                 n_hops=args.n_hops,
                                 feat_fun_flag=args.feat_fun_flag
                                 ) 
    np.save(os.path.join(data_dir,args.feat_fun_flag+'.npy'), feats)

    logger.info("preprocessing ended.")






