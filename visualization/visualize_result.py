'''
Description: 
        1, get all eval_stat.json files;
        2, group by data_name;
        3, group by model; 
        4, group by key_list that to be compared;
version: 
Author: huangzg
LastEditors: huangzg
Date: 2021-08-31 09:59:36
LastEditTime: 2021-10-19 15:07:42
'''
import json
import yaml 
import os
import pandas as pd 

import regex as re
from utils import logger 
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
from typing import Union


parser = argparse.ArgumentParser(description='result visualization')

parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
parser.add_argument('--data_dir',type=str, default='../dataset')
# parser.add_argument('--result_dir',type=str, default='./result')

parser.add_argument('--pub_keys', nargs='*', default=None,
                    help='the pubic keywords for select target files') 

parser.add_argument('--private_keys', nargs='+', default=("model_gat", "model_pgat@use_ori_0"),
                    help='the keywords and/or model_name to be compared, if use keywords, organize them with `@`') 

parser.add_argument('--compared_key', type=str, default='num_layers_',help='the argument key to be compared') 

parser.add_argument('--compared_values', nargs='+', default=list(range(2,7)),
                    help='the values of `compared_key` to select  unique target file') 
parser.add_argument('--loss_file_key', type=str, default='degree_node_graph_model_pgat@@layers_4_use_ori_0_train_losses')

logger.info("start to visualize result...")
args = parser.parse_args()
dataset = DglNodePropPredDataset(name=args.dataset, root=args.data_dir)
split_idx = dataset.get_idx_split()
valid_idx = split_idx["valid"]
test_idx = split_idx["test"]

val_len, test_len = len(valid_idx), len(test_idx) 

data_name = args.dataset.replace("-", "_")


def find_file(data_dir, keyword, full_flag=True):
    """find files in `data_dir` with re """
    pattern = re.compile(keyword)
    if full_flag:
        file_list = [os.path.join(data_dir,file) for file in os.listdir(data_dir) if re.search(pattern, file)]
    
    else:
        file_list = [file for file in os.listdir(data_dir) if re.search(pattern, file)]
        
    return file_list


def read_file(file):
    """read csv、xlsx、json,yaml"""
    if re.search('csv$',file):
        if os.path.getsize(file)/1024**2 > 500:
            chunksize = 200000
        else:
            chunksize = None
        try:
            data = pd.read_csv(file,chunksize=chunksize)
        except UnicodeDecodeError:
            data = pd.read_csv(file,encoding='gb18030',chunksize=chunksize)
        except pd.errors.ParserError:
            data = pd.read_csv(file,sep='\t',encoding='gb18030',chunksize=chunksize)
    elif re.search('xlsx$',file):
        data = pd.read_excel(file)
    elif re.search('json$',file):
        with open(file,'r',encoding='utf-8') as f:
            data = json.load(f)
    elif re.search('(yml|yaml)$',file):
        with open(file,'r',encoding='utf-8') as f:
            data = yaml.load(f)
    else:

        logger.error('wrong file type!')
        return
    return data


def recursive_re(string, reg_list):
    
    match = [True if re.search(reg, string) else False for reg in reg_list]
    
    if all(match):
        return True
    else:
        return False
    

def get_eval_df(data_dir='../result', 
                data_name=data_name, 
                basic_key='eval_stat.json', 
                pub_keys=args.pub_keys, 
                private_keys=args.private_keys,
                compared_key=args.compared_key,
                compared_values=args.compared_values
                ):
    
    # select files by data_name
    files = find_file(data_dir, data_name + ".*" + basic_key)
    # select  files by pub_keys
    files = [file for file in files if recursive_re(file, pub_keys)] if pub_keys else files 
    # final df 
    df = []
    index_list = []
    for private_key in private_keys:
        # for the situation if some cpd_value dose not match any file 
        index_v = []
        # results of one model that contain all cpd_values
        data_private = []
        key_list = private_key.split("@")
        private_files = [file for file in files if recursive_re(file, key_list)] 
        for cpd_value in compared_values:
            unique_file = [file for file in private_files if re.search(compared_key+"_?"+str(cpd_value), file)]
            if not unique_file:
                continue
            index_v.append(cpd_value)
            assert len(unique_file) == 1 
            # only last two values is needed
            data = read_file(unique_file[0])[-2:]
            data_private.append(data)
        index_list.append(index_v)     
        data_private = pd.DataFrame(data_private)
        columns = [private_key+ "_val", private_key + "_test"]
        data_private.columns = columns
        data_private[private_key+'_avg'] = (data_private[columns[0]] * val_len + data_private[columns[1]] * test_len)/(val_len + test_len)
        df.append(data_private)
        
    assert len(index_list[0]) == len(index_list[1])
    df = pd.concat(df, axis=1)
    
    df.index = index_list[0]
    pub_join = "_".join(pub_keys) if pub_keys else ""
    save_name = os.path.join(data_dir, '_'.join([data_name, pub_join, '_'.join(private_keys),compared_key]))
    
    df.to_csv(save_name+'.csv')
    ax = df.plot()
    fig = ax.get_figure()
    fig.savefig(save_name + ".png")
    logger.info("visualize result done!")
    
def plot_train_loss(file_key=args.loss_file_key):
    file_key = file_key.replace("@@",".*")
    file = find_file('../result', file_key,full_flag=True)[0]

    data = read_file(file)
    df = pd.Series(data)
    ax = df.plot()
    fig = ax.get_figure()
    fig.savefig(file.replace('json','png'))
    logger.info("plot train loss done.")
    
if __name__ == "__main__":

    get_eval_df()
    plot_train_loss()
        
    
    