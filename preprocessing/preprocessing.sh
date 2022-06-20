###
 # @Description: 
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-09-09 22:39:51
 # @LastEditTime: 2021-10-19 14:53:48
### 
{

    nohup python -u preprocessing.py --n_hops 3 --to_bidirected 1 \
                                    --use_mp_pre 1 --n_pools 8 \
                                    --feat_fun_flag structure >structure_preprocessing.log 2>&1 &&
    nohup python -u preprocessing.py --n_hops 3 --to_bidirected 1 \
                                    --use_mp_pre 1 --n_pools 8 \
                                    --feat_fun_flag distance >distance_preprocessing.log 2>&1 
} > preprocessing.log 2>&1 &


