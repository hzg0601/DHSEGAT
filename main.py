'''
Description: 
version: 
Author: huangzg
LastEditors: huangzg
Date: 2021-09-18 15:06:38
LastEditTime: 2021-10-15 17:13:43
'''
from args_and_preparation import args, model, model_name, device, labels, g, feats, train_idx, valid_idx, test_idx, evaluator, logger, n_classes, train_pred_idx
from correct_and_smooth import correct_and_smooth, multi_correct_smooth 
import os 

if args.mini_batch:
    from mini_batch_train import train_and_eval
else:
    from train import train_and_eval


def main():
    if args.runs:
        # run target model `runs` times 
        for i in range(args.runs):
            train_and_eval(
                args=args,
                model_base=model,
                model_name_base=model_name,
                g_base=g,
                feats_base=feats,
                labels=labels,
                device=device,
                train_idx=train_pred_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                evaluator=evaluator, 
                n_classes=n_classes,
                run=i)
        multi_correct_smooth(
                args=args,
                model_name_base=model_name,
                g=g,
                feats=feats,
                labels=labels,
                device=device,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                n_classes=n_classes,
                evaluator=evaluator,
                keyword='_run_'
        )
    else:
        # if both of model and eval_stat don't exist or force to retrain, train model 
        if args.retrain or not os.path.exists(os.path.join('./base', model_name+'.pt')):
            logger.info("force to retrain or model/eval_stat does not exist, start to train...")
            train_and_eval(
                args=args,
                model_base=model,
                model_name_base=model_name,
                g_base=g,
                feats_base=feats,
                labels=labels,
                device=device,
                train_idx=train_pred_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                evaluator=evaluator,
                n_classes=n_classes
            )

        else:
            logger.info("skip training, model name:")
            logger.info(model_name)
        correct_and_smooth(
                args=args,
                # model=model,
                model_name=model_name,
                g=g,
                feats=feats,
                labels=labels,
                device=device,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx,
                n_classes=n_classes,
                evaluator=evaluator 
                )
    
        
if __name__ == "__main__":
    main()