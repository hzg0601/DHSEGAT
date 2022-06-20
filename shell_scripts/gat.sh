###
 # @Description: 
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-10-12 13:50:21
 # @LastEditTime: 2021-10-19 15:15:47
### 
{
    
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model gat \
                            --num_layers 3 \
                            --num_heads 3  \
                            --mini_batch 0 \
                            --use_gen 0 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --retrain 1 >> gat_layers_3_heads_3_use_gen_0_gen_dhs.log 2>&1 && 

    
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model gat \
                            --num_layers 4 \
                            --num_heads 3  \
                            --mini_batch 0 \
                            --use_gen 0 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --retrain 1 >> gat_layers_4_heads_3_use_gen_0_gen_dhs.log 2>&1 &&

    
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model gat \
                            --num_layers 5 \
                            --num_heads 3  \
                            --mini_batch 0 \
                            --use_gen 0 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --retrain 1 >> gat_layers_5_heads_3_use_gen_0_gen_dhs.log 2>&1 
}>gat_full_batch.log 2>&1 &
