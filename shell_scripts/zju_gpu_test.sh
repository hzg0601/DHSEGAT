###
 # @Description: 
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-10-19 14:17:11
 # @LastEditTime: 2021-10-19 15:11:39
### 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model gat \
                            --num_layers 2 \
                            --num_heads 3 \
                            --mini_batch 0 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --num_hiddens 256 \
                            --K 3 \
                            --runs 10 \
                            --gpu 0 \
                            --drop 0.5 \
                            --input_drop 0.1 \
                            --edge_drop 0.1 \
                            --attn_drop 0.05 \
                            --scheduler step_add \
                            --optimizer rmsprop \
                            --retrain 1 >gat_ha_layers_2_heads_3_use_gen_1_gen_dhs.log 2>&1 &