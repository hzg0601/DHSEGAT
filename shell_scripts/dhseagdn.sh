###
 # @Description: 
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-10-11 16:30:02
 # @LastEditTime: 2021-10-19 15:15:37
### 
{
    # two layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhseagdn \
                            --num_layers 2 \
                            --num_heads 3 \
                            --mini_batch 0 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --K 3 \
                            --runs 10 \
                            --retrain 1 >>dhseagdn_layers_2_heads_3_use_gen_1_gen_dhs.log 2>&1 &&
    # three layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhseagdn \
                            --num_layers 3 \
                            --num_heads 3 \
                            --mini_batch 0 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --K 3 \
                            --runs 10 \
                            --retrain 1 >>dhseagdn_layers_3_heads_3_use_gen_1_gen_dhs.log 2>&1 &&

    # four layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhseagdn \
                            --num_layers 4 \
                            --num_heads 3 \
                            --mini_batch 0 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --K 3 \
                            --runs 10 \
                            --retrain 1 >>dhseagdn_layers_4_heads_3_use_gen_1_gen_dhs.log 2>&1 &&

    # five layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhseagdn \
                            --num_layers 5 \
                            --num_heads 3  \
                            --mini_batch 0 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --K 3 \
                            --runs 10 \
                            --retrain 1 >>dhseagdn_layers_5_heads_3_use_gen_1_gen_dhs.log 2>&1 

}>dhseagdn_layers.log 2>&1 & 
