
###
 # @Description: 
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-10-13 14:29:09
 # @LastEditTime: 2021-11-04 09:47:31
### 

# encoding layer ablation study 
nohup python -u main.py --dataset ogbn-arxiv  \
                        --model gat-ha \
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
                        --retrain 1 >agdn_layers_3_heads_3_use_gen_1_ablation_study_encoding.log 2>&1 &
                        
nohup python -u main.py --dataset ogbn-arxiv  \
                        --model gat \
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
                        --retrain 1 >gat_layers_3_heads_3_use_gen_1_ablation_study_encoding.log 2>&1 &

# distance ablation study 

nohup python -u main.py --dataset ogbn-arxiv  \
                        --model dhseagdn \
                        --num_layers 3 \
                        --num_heads 3 \
                        --mini_batch 0 \
                        --use_gen 1 \
                        --gen_feature degree node graph \
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
                        --retrain 1 >dhseagdn_layers_3_heads_3_use_gen_1_gen_dhs_ablation_study_distance.log 2>&1 &

nohup python -u main.py --dataset ogbn-arxiv  \
                        --model dhsegat \
                        --num_layers 3 \
                        --num_heads 3 \
                        --mini_batch 0 \
                        --use_gen 1 \
                        --gen_feature degree node graph \
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
                        --retrain 1 >dhsegat_layers_3_heads_3_use_gen_1_gen_dhs_ablation_study_distance.log 2>&1 &

