'''
Description: 
version: 
Author: huangzg
LastEditors: huangzg
Date: 2021-09-30 17:02:40
LastEditTime: 2021-10-15 11:34:16
'''
import copy
import json
import multiprocessing as mp
import os
from typing import Union

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from utils import logger, evaluate, arg_model, optimizer_preparing


def mini_batch_eval(model, valid_dataloader, feats, args, device):
    eval_batch_logits = [] 
    for input_idx, output_idx, block in valid_dataloader:
        batch_feat = feats[input_idx] 
        batch_feat = batch_feat.to(device)
        block = [item.to(device) for item in block] 
         
        logits = arg_model(model, block, batch_feat, args)
        
        eval_batch_logits.append(logits)
    eval_logits = torch.cat(eval_batch_logits,dim=0) 
    
    return eval_logits

def train_and_eval(model_name_base, 
                   device, 
                   args, 
                   model_base, 
                   g_base, 
                   feats_base, 
                   labels, 
                   train_idx, 
                   valid_idx, 
                   test_idx, 
                   evaluator, 
                   n_classes,
                   run=None): 

    logger.info(f'Model parameters: {sum(p.numel() for p in model_base.parameters())}')
    
    # * -----------------------------------------------preparing----------------------- 
    # multiruns 
    if run is not None:
        model_name = model_name_base + f"_run_{run}"
    else:
        model_name = model_name_base  
    logger.info(f'model_name:{model_name}') 
    #  -------to device -------------
    model, g, feats = copy.deepcopy(model_base), copy.deepcopy(g_base), copy.deepcopy(feats_base) 
    model = model.to(device) 
    labels = labels.to(device)

    # ----------sampling batch ----------------
    sampler = MultiLayerFullNeighborSampler(args.num_layers) 
    # ? 
    train_dataloader = NodeDataLoader(g, 
                                    train_idx, 
                                    sampler, 
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=False
                                    ) 
    valid_dataloader = NodeDataLoader(g, 
                                    valid_idx, 
                                    sampler, 
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    drop_last=False
                                    )
    test_dataloader = NodeDataLoader(g,
                                    test_idx,
                                    sampler,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    drop_last=False)  
        
    # ------------------optimizer---------------------- 
    opt, scheduler = optimizer_preparing(model, model_name, args) 
        
    logger.info('---------- Training ----------')
    
    eval_stat = []
    best_acc = 0
    best_model = copy.deepcopy(model)

    # recording every epoch' loss 
    full_train_losses = []           
    for epoch_id in range(args.epochs): 
        train_losses = []   
        model.train()
        for batch_id, (source_ids, target_ids, graph_block) in enumerate(train_dataloader):
            logger.debug(f"the training round epoch --{epoch_id}--, batch --{batch_id}-- start...")
            opt.zero_grad() 
            torch.cuda.empty_cache()
            
            X = feats[source_ids]   
            X = X.to(device)
            graph_block = [block.to(device) for block in graph_block] 
            logits =arg_model(model, graph_block, X, args)
            target_ids = target_ids.to(device)
            train_loss = F.nll_loss(logits, labels.squeeze(1)[target_ids])
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            train_losses += [train_loss.item()]  #get the value of train_losses .cpu().detatch().numpy()  
            logger.debug(" train --epoch %d, batch %d --ended." % (epoch_id, batch_id))
        # compute the final loss of this epoch 
        train_loss = sum(train_losses)  
        logger.debug(" train --epoch %d --ended." % (epoch_id))
        full_train_losses.append(train_loss)  
        #----------------------------eval------------------------------
        model.eval()
        with torch.no_grad():
            logits = mini_batch_eval(model, valid_dataloader, feats, args, device) 
            y_pred = logits.argmax(dim=-1, keepdim=True)
            valid_acc = evaluate(y_pred, labels[valid_idx], None, evaluator)
        if epoch_id % 20 == 0:   
            logger.info(f'''Epoch {epoch_id} | Train loss: {train_loss:.4f} | Valid acc {valid_acc:.4f}''')

        scheduler.step() if args.scheduler !='plateau' else scheduler.step(valid_acc) 
        # ---------------------------save model and test-------------------------
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            logger.info('---------- Testing ----------')
            best_model.eval()
            with torch.no_grad():
                logits = mini_batch_eval(best_model, test_dataloader, feats, args, device) 
            y_pred = logits.argmax(dim=-1, keepdim=True)
            test_acc = evaluate(y_pred, labels[test_idx], None, evaluator)
            logger.info(f'Test acc: {test_acc:.4f}')      
            eval_stat += [valid_acc, test_acc]  
            torch.save(best_model.state_dict(), f'base/{model_name}.pt')
    try:
        train_losses = [float(item) for item in full_train_losses]
        eval_stat = [float(item) for item in eval_stat] 
                   
        with open(os.path.join('./result/', model_name+"_train_losses.json"),'w', encoding='utf-8') as f:
            
            json.dump(train_losses, f)

        with open(os.path.join('./result/', model_name+'_eval_stat.json'),'w',encoding='utf-8') as f:
            json.dump(eval_stat, f)
    except Exception as e:
        logger.error(e)
        logger.error("save statistical data error!") 