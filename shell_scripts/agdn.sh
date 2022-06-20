###
 # @Description: 
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-10-11 16:44:07
 # @LastEditTime: 2021-10-21 17:27:34
### 
# {
#     # three layers 
#     nohup python -u main.py --dataset ogbn-arxiv  \
#                             --model gat-ha \
#                             --num_layers 3 \
#                             --num_heads 3 \
#                             --mini_batch 0 \
#                             --use_gen 0 \
#                             --gen_feature degree node graph shortest_path \
#                             --num_hiddens 256 \
#                             --K 3 \
#                             --runs 10 \
#                             --retrain 1 >>gat_ha_layers_3_heads_3_use_gen_0_gen_dhs.log 2>&1 &&

#     # four layers 
#     nohup python -u main.py --dataset ogbn-arxiv  \
#                             --model gat-ha \
#                             --num_layers 4 \
#                             --num_heads 3 \
#                             --mini_batch 0 \
#                             --use_gen 0 \
#                             --gen_feature degree node graph shortest_path \
#                             --num_hiddens 256 \
#                             --K 3 \
#                             --runs 10 \
#                             --retrain 1 >>gat_ha_layers_4_heads_3_use_gen_0_gen_dhs.log 2>&1 &&

#     # five layers 
#     nohup python -u main.py --dataset ogbn-arxiv  \
#                             --model gat-ha \
#                             --num_layers 5 \
#                             --num_heads 3  \
#                             --mini_batch 0 \
#                             --use_gen 0 \
#                             --gen_feature degree node graph shortest_path \
#                             --num_hiddens 256 \
#                             --K 3 \
#                             --runs 10 \
#                             --retrain 1 >>gat_ha_layers_5_heads_3_use_gen_0_gen_dhs.log 2>&1 
# }>gat_ha_full.log 2>&1 & 
{
    # original_version_of_agdn_acc_7375
nohup python -u main.py --dataset ogbn-arxiv  \
                        --model gat-ha \
                        --num_layers 3 \
                        --num_heads 3 \
                        --mini_batch 0 \
                        --use_gen 0 \
                        --gen_feature degree node graph shortest_path \
                        --num_hiddens 256 \
                        --K 3 \
                        --epochs 2000 \
                        --use_bot 0 \
                        --use_self_kd 0 \
                        --runs 10 \
                        --gpu 1 \
                        --scheduler step_add \
                        --optimizer rmsprop \
                        --attn_drop 0.05 \
                        --input_drop 0.25 \
                        --edge_drop 0.0 \
                        --dropout 0.5 \
                        --use_clip 0 \
                        --model_norm sys \
                        --lr 0.002 \
                        --use_attn_dst 0 \
                        --retrain 1 >agdn_original_version.log 2>&1 &&
    # current_version_of_agdn_0.7346
    # use_attn_dst 1 , but with 0 results better performance; 
    nohup python -u main.py --dataset ogbn-arxiv  \
                        --model gat-ha \
                        --num_layers 3 \
                        --num_heads 3 \
                        --mini_batch 0 \
                        --use_gen 0 \
                        --gen_feature degree node graph shortest_path \
                        --num_hiddens 256 \
                        --K 3 \
                        --epochs 2000 \
                        --use_bot 0 \
                        --use_self_kd 0 \
                        --runs 10 \
                        --gpu 1 \
                        --scheduler step_add \
                        --optimizer rmsprop \
                        --attn_drop 0.0 \
                        --input_drop 0.1 \
                        --edge_drop 0.1 \
                        --dropout 0.75 \
                        --use_clip 0 \
                        --model_norm none \
                        --use_attn_dst 0 \
                        --retrain 1 >agdn_current_version.log 2>&1 
}>agdn_reproducing.log 2>&1 & 

