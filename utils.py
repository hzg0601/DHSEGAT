'''
Description: 
version: 
Author: huangzg
LastEditors: huangzg
Date: 2021-09-18 14:19:48
LastEditTime: 2021-10-29 14:35:05
'''
import logging
import math
import os
import random

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
epsilon = 1 - math.log(2)

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # set logger level

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

def evaluate(y_pred, y_true, idx, evaluator): 
    if y_pred.dim() == 1:
        y_pred = y_pred.squeeze(1)
    if y_true.dim() == 1:
        y_true = y_true.squeeze(1)
        
    assert y_pred.dim() == 2 and y_true.dim() == 2 
    if idx is not None:
        return evaluator.eval({
            'y_true': y_true[idx],
            'y_pred': y_pred[idx]
        })['acc']
    else:
        return evaluator.eval({"y_true":y_true, "y_pred":y_pred})['acc']  
    
    
def gen_feats(g, feats, args, data_dir):
    n_struc = 0  
    if "degree" in args.gen_feature:

        feats = torch.cat([feats, g.in_degrees().unsqueeze(1), g.out_degrees().unsqueeze(1)],axis=1) 
        n_struc += 2 
     
    if "graph" in args.gen_feature:
        gen_data = torch.Tensor(np.stack(np.load(os.path.join(data_dir,"structure"+'.npy'),allow_pickle=True)[:,0],axis=0))
        n_struc += gen_data.size()[1] 
        feats = torch.cat([feats, gen_data], axis=1) 
        
    if "node" in args.gen_feature :
        gen_data = torch.Tensor(np.stack(np.load(os.path.join(data_dir,"structure"+'.npy'),allow_pickle=True)[:,1],axis=0))
        n_struc += gen_data.size()[1]
        feats = torch.cat([feats, gen_data], axis=1)
    n_dist = 0  
    if "shortest_path" in args.gen_feature:
        gen_data = torch.Tensor(np.stack(np.load(os.path.join(data_dir,"distance"+'.npy'),allow_pickle=True),axis=0))
        n_dist += gen_data.size()[1] 
        feats = torch.cat([feats, gen_data], axis=1)
        
    return feats, n_struc, n_dist 


def cross_entropy(x, labels):
    return F.cross_entropy(x, labels[:, 0], reduction="mean")

def loge_cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def self_kd(all_out,teacher_all_out,temperature, ori_loss, kd_alpha):
    T = temperature
    kd_loss = torch.nn.KLDivLoss()(F.log_softmax(all_out/T, dim=1), F.softmax(teacher_all_out/T, dim=1)) * (T * T)
    self_kd_loss = ori_loss * (1-kd_alpha) + kd_loss * kd_alpha
    return self_kd_loss


def bot(n_label_iters, model, graph, feat, unlabel_idx, n_classes, pred):
    
    for _ in range(n_label_iters):
        pred = pred.detach()
        torch.cuda.empty_cache()
        feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
        pred = model(graph, feat)
        
    return pred

def compute_norm(graph):
    degs = graph.in_degrees().float().clamp(min=1)
    deg_isqrt = torch.pow(degs, -0.5)

    degs = graph.in_degrees().float().clamp(min=1)
    deg_sqrt = torch.pow(degs, 0.5)

    return deg_sqrt, deg_isqrt


def norm_graph(graph):
    deg_sqrt, deg_isqrt = compute_norm(graph)
    
    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_isqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))

    graph.srcdata.update({"src_norm": deg_isqrt})
    graph.dstdata.update({"dst_norm": deg_sqrt})
    graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))
    
    return graph 


def add_labels(feat, labels, idx, n_classes, device='cpu'):
    onehot = torch.zeros([feat.shape[0], n_classes])
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot.to(device)], dim=-1)


def arg_model(model, g, feat, args):
    if 'gat' in args.model or 'agdn' in args.model:
        return model(g, feat)
    else:
        return model(feat)


class StepAddScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, init_lr=0.01, threshold_epoch=50, last_epoch=-1,verbose=False) -> None:
        self.threshold_epoch = threshold_epoch
        self.init_lr = init_lr 
        super(StepAddScheduler,self).__init__(optimizer, last_epoch, verbose)
  
    def get_lr(self):
        if 0 < self.last_epoch <= self.threshold_epoch:
            return [self.init_lr * self.last_epoch/self.threshold_epoch for group in self.optimizer.param_groups]
            
        elif self.last_epoch > self.threshold_epoch:
            return [self.init_lr for group in self.optimizer.param_groups]
        
        else: # -1
            return [self.init_lr/self.threshold_epoch for group in self.optimizer.param_groups]  
        
    
def optimizer_preparing(model, model_name_run, args):
    if args.use_checkpoint and os.path.exists(f'base/{model_name_run}.pt'):
        model.load_state_dict(torch.load(f'base/{model_name_run}.pt')) 
        last_epoch = args.use_checkpoint
    else:
        last_epoch = -1 
    if args.optimizer == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=args.lr) if last_epoch <= 0 else optim.RMSprop([{"params":model.parameters(), "initial_lr":args.lr}], lr=args.lr)
    elif args.optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr) if last_epoch <= 0 else optim.Adam([{"params":model.parameters(), "initial_lr":args.lr}], lr=args.lr)
    # --------scheduler----------------------------
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50, eta_min=1e-7, last_epoch=last_epoch)
    # ? step add to skip local minimum?
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.8, patience=5, min_lr=1e-7)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.9,last_epoch=last_epoch)
    elif args.scheduler == 'step_add':
        scheduler = StepAddScheduler(opt, init_lr=args.lr, threshold_epoch=50, last_epoch=last_epoch) 
    else:
        logger.error("undefined scheduler type!")
        raise KeyError 
    return opt, scheduler


def rename_file(path='./result', old_keyword='', new_keyword=""):
    
    files = os.listdir(path)
    for file in files:
        old = os.path.join(path, file)
        new = os.path.join(path, file.replace(old_keyword, new_keyword))
        os.rename(old, new)
    print("rename file done!")
    