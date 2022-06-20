'''
Description: 
version: 
Author: huangzg
LastEditors: huangzg
Date: 2021-08-22 16:43:55
LastEditTime: 2021-10-29 14:31:28
'''
from types import new_class
import torch
import re  
import os 
from copy import copy 
import numpy as np 
from model import CorrectAndSmooth
from utils import logger, evaluate
import torch.nn.functional as F 

def eval_cs(mask_idx, eval_idx, labels, y_soft, g, args, evaluator):
    
    cs = CorrectAndSmooth(num_correction_layers=args.num_correction_layers,
                            correction_alpha=args.correction_alpha,
                            correction_adj=args.correction_adj,
                            num_smoothing_layers=args.num_smoothing_layers,
                            smoothing_alpha=args.smoothing_alpha,
                            smoothing_adj=args.smoothing_adj,
                            scale=args.scale)
    
    if args.use_correct:
        y_soft = cs.correct(g, y_soft, labels[mask_idx], mask_idx)
    y_soft = cs.smooth(g, y_soft, labels[mask_idx], mask_idx)
    y_pred = y_soft.argmax(dim=-1, keepdim=True)
    acc_cs = evaluate(y_pred, labels, eval_idx, evaluator)
    
    return acc_cs 


def correct_and_smooth(model_name, args, g, feats, labels, train_idx, valid_idx, test_idx, n_classes, evaluator, device="cpu"):

    # model = model.to(device)
    g = g.to(device)
    feats = feats.to(device) 
    labels = labels.to(device)
    train_idx, valid_idx, test_idx =train_idx.to(device), valid_idx.to(device), test_idx.to(device)

    logger.info("the model in correct and smooth is: ")
    logger.info(model_name)

    logger.info('---------- Before ----------')
    y_soft = torch.load(f'result/{model_name}_best_logits.pt').softmax(dim=-1).to(device)
    if args.use_bot:
        y_soft[train_idx] = F.one_hot(labels[train_idx], n_classes).float().squeeze(1) 
    # else:
    #     model.load_state_dict(torch.load(f'base/{model_name}.pt'))
    #     model.eval()
    #     with torch.no_grad():
    #         y_soft = arg_model(model, g, feats, args).softmax(dim=-1) # for cross_entropy loss, softmaxed logits
    #         # y_soft = arg_model(model, g, feats, args).exp() # for nll_loss 
    y_pred = y_soft.argmax(dim=-1,keepdim=True)
    valid_acc_base = evaluate(y_pred, labels, valid_idx, evaluator)
    test_acc_base = evaluate(y_pred, labels, test_idx, evaluator)
    logger.info(f'Valid acc: {valid_acc_base:.4f} | Test acc: {test_acc_base:.4f}')

    logger.info('---------- Correct & Smoothing ----------')
    
    valid_acc_cs = eval_cs(mask_idx=train_idx, eval_idx=valid_idx,labels=labels,y_soft=y_soft,g=g,args=args,evaluator=evaluator)
    
    mask_idx = torch.cat([train_idx, valid_idx])
    test_acc_cs = eval_cs(mask_idx=mask_idx, eval_idx=test_idx,labels=labels,y_soft=y_soft,g=g,args=args,evaluator=evaluator)
    logger.info(f'Valid acc C&S: {valid_acc_cs:.4f} | Test acc C&S: {test_acc_cs:.4f}')
    
    return valid_acc_base, test_acc_base, valid_acc_cs, test_acc_cs
    
    
def multi_correct_smooth(model_name_base, args, g, feats, labels, train_idx, valid_idx, test_idx, n_classes, evaluator, keyword='_run_', device="cpu"):
    
    files = [file for file in os.listdir('./base') if re.search(model_name_base+keyword, file)] 
    
    valid_base, test_base, valid_cs, test_cs = [], [], [], []
    for file in files:
        model_name = file.replace(".pt", '')
        valid_acc_base, test_acc_base, valid_acc_cs, test_acc_cs = correct_and_smooth(model_name=model_name, 
                                                                                      args=args,
                                                                                      g=g,
                                                                                      feats=feats,
                                                                                      labels=labels,
                                                                                      train_idx=train_idx,
                                                                                      valid_idx=valid_idx,
                                                                                      test_idx=test_idx,
                                                                                      n_classes=n_classes,
                                                                                      evaluator=evaluator,
                                                                                      device=device)
        valid_base.append(valid_acc_base)
        test_base.append(test_acc_base) 
        valid_cs.append(valid_acc_cs) 
        test_cs.append(test_acc_cs)
    valid_base_mean, valid_base_std = np.mean(valid_base), np.std(valid_base)
    test_base_mean, test_base_std = np.mean(test_base), np.std(test_base)
    valid_cs_mean, valid_cs_std = np.mean(valid_cs), np.std(valid_cs)
    test_cs_mean, test_cs_std = np.mean(test_cs), np.std(test_cs)
    logger.info(f'Valid acc Base: {valid_base_mean:.4f} +/- {valid_base_std:.4f}| Test acc Base: {test_base_mean:.4f} +/- {test_base_std:.4f}')
    logger.info(f'Valid acc C&S: {valid_cs_mean:.4f} +/- {valid_cs_std:.4f}| Test acc C&S: {test_cs_mean:.4f} +/- {test_cs_std:.4f}')
    


        