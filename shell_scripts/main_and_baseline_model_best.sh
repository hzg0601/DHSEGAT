###
 # @Description: main_model_and_baseline shell scripts
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-10-13 14:29:09
 # @LastEditTime: 2021-11-04 09:46:34
### 
# {
# dhseagdn 
nohup python -u main.py --dataset ogbn-arxiv  \
                        --model dhseagdn \
                        --num_layers 3 \
                        --num_heads 3 \
                        --mini_batch 0 \
                        --use_gen 1 \
                        --gen_feature degree node graph shortest_path \
                        --num_hiddens 256 \
                        --K 3 \
                        --epochs 3000 \
                        --use_bot 0 \
                        --use_self_kd 0 \
                        --runs 10 \
                        --gpu 0 \
                        --scheduler cosine \
                        --optimizer adam \
                        --attn_drop 0.0 \
                        --input_drop 0.1 \
                        --edge_drop 0.1 \
                        --dropout 0.5 \
                        --use_clip 0 \
                        --use_attn_dst 0 \
                        --use_correct 1 \
                        --model_norm sys \
                        --retrain 1 >dhseagdn_layers_3_heads_3_use_gen_1_gen_dhs_use_bot_0_use_self_kd_0.log 2>&1 &
# * repeat baseline gat model, do not use gen features
nohup python -u main.py --dataset ogbn-arxiv  \
                        --model gat \
                        --num_layers 3 \
                        --num_heads 3 \
                        --mini_batch 0 \
                        --use_gen 0 \
                        --gen_feature degree node graph shortest_path \
                        --num_hiddens 256 \
                        --K 3 \
                        --epochs 3000 \
                        --use_bot 0 \
                        --use_self_kd 0 \
                        --runs 10 \
                        --gpu 0 \
                        --scheduler cosine \
                        --optimizer adam \
                        --attn_drop 0.0 \
                        --input_drop 0.1 \
                        --edge_drop 0.1 \
                        --dropout 0.5 \
                        --use_clip 0 \
                        --use_attn_dst 0 \
                        --use_correct 1 \
                        --model_norm sys \
                        --retrain 1 >gat_layers_3_heads_3_use_gen_0_gen_dhs_use_bot_0_use_self_kd_0.log 2>&1 &
# * repeate baseline agdn model, do not use gen features
nohup python -u main.py --dataset ogbn-arxiv  \
                        --model gat-ha \
                        --num_layers 3 \
                        --num_heads 3 \
                        --mini_batch 0 \
                        --use_gen 0 \
                        --gen_feature degree node graph shortest_path \
                        --num_hiddens 256 \
                        --K 3 \
                        --epochs 3000 \
                        --use_bot 0 \
                        --use_self_kd 0 \
                        --runs 10 \
                        --gpu 1 \
                        --scheduler cosine \
                        --optimizer adam \
                        --attn_drop 0.0 \
                        --input_drop 0.1 \
                        --edge_drop 0.1 \
                        --dropout 0.5 \
                        --use_clip 0 \
                        --use_attn_dst 0 \
                        --use_correct 1 \
                        --model_norm sys \
                        --retrain 1 >agdn_layers_3_heads_3_use_gen_0_gen_dhs_use_bot_0_use_self_kd_0.log 2>&1 &
# dhsegat 
nohup python -u main.py --dataset ogbn-arxiv  \
                        --model dhsegat \
                        --num_layers 3 \
                        --num_heads 3 \
                        --mini_batch 0 \
                        --use_gen 1 \
                        --gen_feature degree node graph shortest_path \
                        --num_hiddens 256 \
                        --K 3 \
                        --epochs 3000 \
                        --use_bot 0 \
                        --use_self_kd 0 \
                        --runs 10 \
                        --gpu 1 \
                        --scheduler cosine \
                        --optimizer adam \
                        --attn_drop 0.0 \
                        --input_drop 0.1 \
                        --edge_drop 0.1 \
                        --dropout 0.5 \
                        --use_clip 0 \
                        --use_attn_dst 0 \
                        --use_correct 1 \
                        --model_norm sys \
                        --retrain 1 >dhsegat_layers_3_heads_3_use_gen_1_gen_dhs_use_bot_0_use_self_kd_0.log 2>&1 &


                            
# }>main_model_and_baseline.log 2>&1 & 