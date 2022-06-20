
###
 # @Description: 
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-08-22 16:43:55
 # @LastEditTime: 2021-10-19 15:15:56
### `huangzg_sggd`  conda virtual env;
{   
    # three layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhsegat \
                            --num_layers 3 \
                            --num_heads 4 \
                            --use_ori 0  \
                            --mini_batch 1 --epochs 250 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --batch_size 512 \
                            --num_hiddens 256 \
                            --retrain 1 >> dhsegat_layers_3_heads_4_use_gen_1_gen_dhs.log 2>&1 &&
    # four layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhsegat \
                            --num_layers 4 \
                            --num_heads 4 \
                            --use_ori 0  \
                            --mini_batch 1 --epochs 250 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --batch_size 512 \
                            --num_hiddens 256 \
                            --retrain 1 >> dhsegat_layers_4_heads_4_use_gen_1_gen_dhs.log 2>&1 &&
    # five layers 
    nohup python -u main.py --dataset ogbn-arxiv  \
                            --model dhsegat \
                            --num_layers 5 \
                            --num_heads 4 \
                            --use_ori 0  \
                            --mini_batch 1 --epochs 250 \
                            --use_gen 1 \
                            --gen_feature degree node graph shortest_path \
                            --batch_size 512 \
                            --num_hiddens 256 \
                            --retrain 1 >> dhsegat_layers_5_heads_4_use_gen_1_gen_dhs.log 2>&1 &&

} >dhsegat_mini_batch.log 2>&1 & 


