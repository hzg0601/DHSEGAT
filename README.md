<!--
 * @Description: 
 * @version: 
 * @Author: huangzg
 * @LastEditors: huangzg
 * @Date: 2021-10-19 15:16:27
 * @LastEditTime: 2021-10-20 14:22:18
-->

1, to run preprocessing to generate distance and hop-wise structure feature[^1]:

```
cd preprocessing
bash preprocessing.sh
```

2, to reproduce main_model_and baseline model[^2]

```
bash shell_scripts/main_and_baseline_model.sh

```

3, to do ablation study, run following shell scripts in turns:

```

bash shell_scripts/distance_and_encoding_layer_ablation_study.sh
bash shell_scripts/degree_and_node_ablation_study.sh
bash shell_scripts/graph_ablation_study.sh
```

```

```
[^1]: The scripts are written on windows, if one want to run on linux, run following code in turns:

        ```
        
        vim shell_scripts/target-script.sh
        
        :set fileformat=unix
        
        :wq
        
        ```
[^2]: the shell scripts of main_and_baseline_model.sh and ablation study model are supposed to run on 2 gpus.
