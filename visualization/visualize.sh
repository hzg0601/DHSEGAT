###
 # @Description: 
 # @version: 
 # @Author: huangzg
 # @LastEditors: huangzg
 # @Date: 2021-09-24 10:00:25
 # @LastEditTime: 2021-10-19 15:06:04
### 
{
        # visualize result 
    nohup python -u visualize_result.py --dataset ogbn-arxiv \
                                        --private_keys model_gat@degree_node_graph_mode model_pgat@degree_node_graph_mode \
                                        --compared_key num_layers_ \
                                        --compared_values 2 3 4 5 6 >>ogbn_arxiv_layers_use_gen_1_result_gen_dngr.log 2>&1 
    nohup python -u visualize_result.py --dataset ogbn-arxiv \
                                        --private_keys arxiv_model_gat arxiv_model_pgat@use_ori_0 \
                                        --compared_key num_layers_ \
                                        --compared_values 2 3 4 5 6 >>ogbn_arxiv_layers_use_ori_1_result_gen_dngr.log 2>&1
}
