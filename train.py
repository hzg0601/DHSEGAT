import copy
import json
import os 
import torch
import torch.nn.functional as F
from utils import logger, evaluate, bot, self_kd, arg_model, optimizer_preparing


# --------------train and eval ------------------------------------------------------
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
    # -----------------------preparing----------------------- 
    # ------multiruns ------
    model_name = model_name_base + f"_run_{run}" if run is not None else model_name_base
    logger.info(f'model_name:{model_name}') 
    model, g, feats = copy.deepcopy(model_base), copy.deepcopy(g_base), copy.deepcopy(feats_base)
    opt, scheduler = optimizer_preparing(model, model_name, args) 
    #  -------to device -------------
    model, labels, g, feats = model.to(device), labels.to(device), g.to(device), feats.to(device)
    train_idx, valid_idx, test_idx = train_idx.to(device), valid_idx.to(device), test_idx.to(device)

    logger.info('---------- Training ----------')
    
    eval_stat = []
    best_acc = 0
    best_model = copy.deepcopy(model) 
    # to save current training's result
    best_logits_dir = os.path.join('./result/', model_name+"_best_logits.pt") 
    # load last training's result for self-kd 
    teacher_logits_dir = best_logits_dir.replace(str(run), str(run-1)) if run is not None and run != 0 else best_logits_dir.replace('_run_0','') 
    best_logits_teacher = torch.load(teacher_logits_dir).to(device) if os.path.exists(teacher_logits_dir) else None 
    full_train_losses = []           
    for epoch_id in range(args.epochs):
        torch.cuda.empty_cache() 
        train_losses = []   
        model.train()
        opt.zero_grad() 
        logits = arg_model(model, g, feats, args) # un-softmax-ed logits 
        # -----bot--------------
        if args.use_bot:
            unlabel_idx = torch.cat([train_idx, valid_idx, test_idx])
            logits = bot(int(args.n_label_iters), model, g, feats, unlabel_idx, n_classes, logits) # will update feats for every epoch
        train_loss = F.cross_entropy(logits[train_idx], labels.squeeze(1)[train_idx])
        # train_loss = F.nll_loss(logits[train_idx], labels.squeeze(1)[train_idx]) 
        # ----self_kd------------ 
        if args.use_self_kd and best_logits_teacher is not None:
            train_loss = self_kd(logits, best_logits_teacher, args.temp, train_loss, args.kd_alpha)                
        train_loss.backward()  
        if args.use_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)                 
        opt.step()

        train_loss = train_loss.item() 
        logger.debug(" train --epoch %d --ended." % (epoch_id))
        full_train_losses.append(train_loss)  

        # ------------------------eval------------------------------
        model.eval()
        with torch.no_grad():                               
            logits = arg_model(model, g, feats, args)
            y_pred = logits.argmax(dim=-1, keepdim=True)
            valid_acc = evaluate(y_pred, labels, valid_idx, evaluator)
            eval_stat += [valid_acc]
        if epoch_id % 50 == 0:   
            logger.info(f'''Epoch {epoch_id} | Train loss: {train_loss:.4f} | Valid acc {valid_acc:.4f}''')

        scheduler.step() if args.scheduler !='plateau' else scheduler.step(valid_acc) 
        # ---------------------------save model and test-------------------------
        if valid_acc > best_acc: # todo maybe with valid_loss derivate better performance?
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            best_logits = copy.deepcopy(logits)
            best_pred = copy.deepcopy(y_pred)
            logger.info(f"Epoch {epoch_id} Updated!!! | Best Valid acc:{best_acc:.4f}")
            
    logger.info('---------- Testing ----------')
    
    torch.save(best_model.state_dict(), f'base/{model_name}.pt') 
     
    torch.save(best_logits, best_logits_dir) 
    
    test_acc = evaluate(best_pred, labels, test_idx, evaluator) 
    final_test_acc = evaluate(y_pred, labels, test_idx, evaluator)
    logger.info(f'Best Valid acc:{best_acc:.4f}|Best Test acc: {test_acc:.4f}|Final Test acc: {final_test_acc:.4f}') 
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
               
    

