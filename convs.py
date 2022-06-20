'''
Description: 
version: 
Author: huangzg
LastEditors: huangzg
Date: 2021-10-08 15:06:19
LastEditTime: 2022-05-18 16:48:58
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

# fork from @skepsun
class GATHAConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        K=3,
        num_heads=3,
        feat_drop=0.0,
        edge_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        norm='sym'
    ):
        super(GATHAConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._K = K
        self._norm = norm

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.position_emb = nn.Parameter(torch.FloatTensor(size=(K+1, num_heads, out_feats)))
        self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.hop_attn_bias_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 1)))
        self.hop_attn_bias_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 1)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False) # 
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.position_emb)
        nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
        nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        nn.init.xavier_normal_(self.hop_attn_bias_l, gain=gain)
        nn.init.xavier_normal_(self.hop_attn_bias_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src
            
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=graph.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
            else:
                eids = torch.arange(graph.number_of_edges(), device=graph.device)

            el = (feat_src * self.attn_l).sum(-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            
            # compute softmax
            
            graph.edata["a"] = torch.zeros_like(e)
            graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            shp = graph.edata["gcn_norm"].shape + (1,) * (feat_dst.dim() - 1)
            if self._norm == "sym":
                graph.edata["a"][eids] = graph.edata["a"][eids] * torch.reshape(graph.edata["gcn_norm_adjust"], shp)[eids]
            if self._norm == "avg":
                graph.edata["a"][eids] = (graph.edata["a"][eids] + torch.reshape(graph.edata["gcn_norm"],shp)[eids])/2
            
            hstack = [graph.dstdata["ft"]]

            for _ in range(self._K):
                # message passing, take attention matrix as transation matrix
                graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
                # computes a message on an edge by performing element-wise mul between features of u and e if the features have the same shape; 
                # otherwise, it first broadcasts the features to a new shape and performs the element-wise operation
                hstack.append(graph.dstdata["ft"])
  
            hstack = [h + self.position_emb[[k], :, :] for k, h in enumerate(hstack)]
            a_l = (hstack[0] * self.hop_attn_l).sum(dim=-1).unsqueeze(-1)
            astack_r = [(feat_dst * self.hop_attn_r).sum(dim=-1).unsqueeze(-1) for feat_dst in hstack]
            a = torch.cat([(a_r + a_l) for a_r in astack_r], dim=-1)
            # a = torch.sigmoid(a)
            a = self.leaky_relu(a)
            a = F.softmax(a, dim=-1)
            a = self.attn_drop(a)
            # a = F.dropout(a, p=0.5, training=self.training)
            rst = 0
            for i in range(a.shape[-1]):
                rst += hstack[i] * a[:, :, [i]]

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(feat).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            return rst