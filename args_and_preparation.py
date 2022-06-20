import argparse
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import torch.nn.functional as F
from model import GAT, MLP, MLPLinear, DHSEGAT, GATHA, DHSEAGDN 
import torch
import dgl 
from typing import Union, Optional
from utils import logger, gen_feats, add_labels, norm_graph 
import os 
import re 
parser = argparse.ArgumentParser(description='Base predictor(C&S)')

# Dataset
parser.add_argument('--gpu', type=int, default=0, help='-1 for cpu')
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', choices=['ogbn-arxiv', 'ogbn-products'])
parser.add_argument("--data_dir", type=str, default='dataset')

# training settings
parser.add_argument("--mini_batch", type=int, default=0, help="implement mini-batch training or not")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument("--use_checkpoint", type=int, default=0, 
                    help='do not load checkpoint when 0, otherwise denotes the last epoch of scheduler')
# * scheduler, default to use step_add  
parser.add_argument('--scheduler', type=str, default='cosine', choices=['step','cosine','plateau', 'step_add'])
parser.add_argument('--retrain', type=int, default=1, help='force to retrain even if model already exists')

parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--runs', type=int, default=None, help='num of training runs')

# preprocessing features before training 
parser.add_argument('--use_generated', type=int, default=1, help="use generated feature or not")
parser.add_argument('--use_norm', type=int, default=0, help="normalizing or not for original node feature")
parser.add_argument("--gen_feature",nargs='+', default=["degree",'node','graph', "shortest_path"], help='generated feature type to use')

# Base predictor
parser.add_argument('--model', type=str, default='dhseagdn', choices=['mlp', 'linear', 'gat','dhsegat', 'gat-ha', 'dhseagdn'])
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_hiddens', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--use_clip', type=int, default=0, choices=[0,1])
parser.add_argument('--clip',type=float, default=0.25, help='gradient norm clipping')
# 20211019, add argument optimizer and set default to adam
parser.add_argument('--optimizer', type=str, default='adam')
# * extra options for gat and agdn 
parser.add_argument('--num_heads', type=int, default=3)
parser.add_argument('--input_drop', type=float, default=0.1)
parser.add_argument('--attn_drop', type=float, default=0.0)
parser.add_argument('--edge_drop', type=float, default=0.1)

#extra options for gat-ha 
parser.add_argument('--use_attn_dst',type=int, default=0)
parser.add_argument('--model_norm', type=str, default='sys')
parser.add_argument('--K', type=int, default=3)
# BoT
parser.add_argument('--use_bot',default=0, type=int)
parser.add_argument('--mask_rate', default=0.7, type=float)
parser.add_argument('--n_label_iters', default=1, type=int, help='the num of label iters in BoT')
# self-KD
parser.add_argument('--use_self_kd', default=0, type=int)
parser.add_argument('--temp', default=0.7, type=float)
parser.add_argument('--kd_alpha', default=0.9, type=float)
# C & S
parser.add_argument('--use_correct', type=int, default=1) 
parser.add_argument('--num_correction_layers', type=int, default=50)
parser.add_argument('--correction_alpha', type=float, default=0.979)
parser.add_argument('--correction_adj', type=str, default='DAD')
parser.add_argument('--num_smoothing_layers', type=int, default=50)
parser.add_argument('--smoothing_alpha', type=float, default=0.756)
parser.add_argument('--smoothing_adj', type=str, default='DAD')
parser.add_argument('--scale', type=float, default=20)

args = parser.parse_args()
logger.info(args)
logger.info("start args_and_preparation...") 
# ------------------------------global settting ------------------------------    
device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
# load data
dataset = DglNodePropPredDataset(name=args.dataset, root=args.data_dir)
data_dir = os.path.join(args.data_dir, args.dataset.replace("-", "_"), 'processed')
print(data_dir)
evaluator = Evaluator(name=args.dataset)

split_idx = dataset.get_idx_split()
g, labels = dataset[0]
feats = g.ndata['feat'] 
train_idx = split_idx["train"]
valid_idx = split_idx["valid"]
test_idx = split_idx["test"]

n_features = feats.size()[-1]
n_classes = dataset.num_classes

# -------------make sure the dir to save model parameter exists--------
if not os.path.exists('base'):
    os.makedirs('base')
    
if not os.path.exists('result'):
    os.makedirs('result')

# ----------features preprocessing-------------------------
if args.use_norm:
    feats = (feats - feats.mean(0)) / feats.std(0)
    
if args.use_generated:
    processed_data_name = '_'.join([args.dataset.replace('-','_'),
                            "_".join(args.gen_feature)
                            ])
    
    feats, n_struc, n_dist = gen_feats(g, feats, args, data_dir)
    n_features = feats.size()[1] 
else:
    processed_data_name = args.dataset.replace('-','_')
    
# # ----use_bot,use label features---------   
if args.use_bot:
    mask = torch.rand(train_idx.shape) < args.mask_rate
    train_labels_idx = train_idx[mask]
    feats = add_labels(feats, labels, train_labels_idx, n_classes)
    train_pred_idx = train_idx[~mask]
    assert int(args.n_label_iters) > 0 
    n_features += n_classes
else:
    train_pred_idx = train_idx

# define model          
if args.model == "dhsegat": 
    n_features = n_features - n_struc - n_dist
    model = DHSEGAT(in_feats=n_features,
                 n_classes=n_classes,
                 num_hiddens=args.num_hiddens,
                 num_layers=args.num_layers,
                 num_heads=args.num_heads,
                 activation=F.relu,
                 dropout=args.dropout,
                 attn_drop=args.attn_drop,
                 n_struc=n_struc,
                 n_dist=n_dist
                 )     
    model_name = '_'.join([processed_data_name,
                            "model_" + args.model,
                            "num_heads_" + str(args.num_heads),
                            "num_layers_" + str(args.num_layers)
                            ])
    
elif args.model == "dhseagdn":
    n_features = n_features - n_struc - n_dist
    model = DHSEAGDN(in_feats=n_features,
                n_classes=n_classes,
                num_hiddens=args.num_hiddens,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                activation=F.relu,
                K=args.K,
                dropout=args.dropout,
                attn_drop=args.attn_drop,
                input_drop=args.input_drop,
                edge_drop=args.edge_drop,
                use_attn_dst=args.use_attn_dst,
                norm=args.model_norm,
                n_struc=n_struc,
                n_dist=n_dist,
                 )     
    model_name = '_'.join([processed_data_name,
                            "model_" + args.model,
                            "num_heads_" + str(args.num_heads),
                            "num_layers_" + str(args.num_layers),
                            "K_" + str(args.K)
                            ])
    
elif args.model == 'gat':
    model = GAT(in_feats=n_features,
                n_classes=n_classes,
                num_hiddens=args.num_hiddens,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                activation=F.relu,
                dropout=args.dropout,
                attn_drop=args.attn_drop)
    model_name = '_'.join([processed_data_name,                        
                        "model_" + args.model,
                        "num_heads_" + str(args.num_heads),
                        "num_layers_" + str(args.num_layers)
                        ])
elif args.model == "gat-ha":
    model = GATHA(
                in_feats=n_features,
                n_classes = n_classes,
                K=args.K,
                num_hiddens=args.num_hiddens,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                activation=F.relu,
                dropout=args.dropout,
                input_drop=args.input_drop,
                edge_drop=args.edge_drop,
                attn_drop=args.attn_drop,
                use_attn_dst=args.use_attn_dst,
                norm=args.model_norm,
                )
    model_name = '_'.join([processed_data_name,                        
                        "model_" + args.model,
                        "num_heads_" + str(args.num_heads),
                        "num_layers_" + str(args.num_layers),
                        "K_"+str(args.K) 
                        ])
    
elif args.model == 'mlp':
    model = MLP(n_features, args.num_hiddens, n_classes, args.num_layers, args.dropout)
    model_name = '_'.join([processed_data_name,
                    "model_" + args.model,
                    "num_layers_" + str(args.num_layers)
                    ])
    
elif args.model == "linear":
    model = MLPLinear(n_features, n_classes)
    model_name = '_'.join([processed_data_name,
                "model_" + args.model
                ])
else:
    raise NotImplementedError(f'Model {args.model} is not supported.')    
# ----------preprocess graph---------------------
if args.dataset == 'ogbn-arxiv':
    if 'gat' in args.model or 'agdn' in args.model: 
        g = dgl.add_reverse_edges(g, copy_ndata=True)
        g = g.add_self_loop()
    else:
        g = dgl.to_bidirected(g, copy_ndata=True) 
g = norm_graph(g) if re.search(r'gat[\-_]?ha|agdn', model_name) else g
del dataset        
    
logger.info("args_and_preparation done!")   
     



 
    


