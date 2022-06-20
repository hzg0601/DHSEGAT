
###
 # @Description: 
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-08-22 16:43:55
 # @LastEditTime: 2021-10-19 15:15:40
### `huangzg_sggd`  conda virtual env;
{   
    # two layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhsegat \
                            --num_layers 2 \
                            --num_heads 3 \
                            --mini_batch 0 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --runs 10 \
                            --retrain 1 >> dhsegat_layers_2_heads_3_use_gen_1_gen_dhs.log 2>&1 &&
    # three layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhsegat \
                            --num_layers 3 \
                            --num_heads 3 \
                            --mini_batch 0 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --runs 10 \
                            --retrain 1 >> dhsegat_layers_3_heads_3_use_gen_1_gen_dhs.log 2>&1 &&

    # four layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhsegat \
                            --num_layers 4 \
                            --num_heads 3 \
                            --mini_batch 0 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --runs 10 \
                            --retrain 1 >> dhsegat_layers_4_heads_3_use_gen_1_gen_dhs.log 2>&1 &&

    # five layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhsegat \
                            --num_layers 5 \
                            --num_heads 3  \
                            --mini_batch 0 \
                            --use_gen 1 \
                            --runs 10 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --retrain 1 >> dhsegat_layers_5_heads_3_use_gen_1_gen_dhs.log 2>&1 &&


} >dhsegat_layers.log 2>&1 & 


