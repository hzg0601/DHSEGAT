import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from convs import GATHAConv 
class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)  
        # return F.log_softmax(self.linear(x), dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, num_hiddens, out_dim, num_layers, dropout=0.):
        super(MLP, self).__init__()
        assert num_layers >= 2

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linears.append(nn.Linear(in_dim, num_hiddens))
        self.bns.append(nn.BatchNorm1d(num_hiddens))

        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(num_hiddens, num_hiddens))
            self.bns.append(nn.BatchNorm1d(num_hiddens))
        
        self.linears.append(nn.Linear(num_hiddens, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()

    def forward(self, x):
        for linear, bn in zip(self.linears[:-1], self.bns):
            x = linear(x) 
            x = F.relu(x, inplace=True)
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        return x 
        # return F.log_softmax(x, dim=-1)
    
# implementation from @Espylapiza
class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x
    
class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.bias = nn.Parameter(torch.Tensor(size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x + self.bias



class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 num_hiddens,
                 num_layers,
                 num_heads,
                 activation,
                 dropout=0.0,
                 attn_drop=0.0):
        super().__init__()
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_hidden = num_heads * num_hiddens if i > 0 else in_feats
            out_hidden = num_hiddens if i < num_layers - 1 else n_classes
            out_channels = num_heads

            self.convs.append(GATConv(in_hidden,
                                      out_hidden,
                                      num_heads=num_heads,
                                      attn_drop=attn_drop))
            self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

        # self.bias_last = Bias(n_classes)
        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.dropout0(h)
        for i in range(self.num_layers):
            # for mini-batch training 
            if isinstance(graph, list):
                conv = self.convs[i](graph[i], h)
                linear = self.linear[i](h[:graph[i].num_dst_nodes()]).view(conv.shape)
            else:
                conv = self.convs[i](graph, h)
                linear = self.linear[i](h).view(conv.shape)

            h = conv + linear

            if i < self.num_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)
        
        h = h.mean(1)
        h = self.bias_last(h)
        return h 
        # return F.log_softmax(h, dim=-1)

class GATHA(nn.Module):
    def __init__(
        self, 
        in_feats, 
        n_classes, 
        num_hiddens, 
        num_layers, 
        num_heads, 
        activation, 
        K=3, 
        dropout=0.0, 
        input_drop=0.0, 
        edge_drop=0.0, 
        attn_drop=0.05, 
        use_attn_dst=True,
        norm='both'
    ):
        super().__init__()
        self.in_feats = in_feats
        self.num_hiddens = num_hiddens
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_hidden = num_heads * num_hiddens if i > 0 else in_feats
            out_hidden = num_hiddens if i < num_layers - 1 else n_classes
            # in_channels = num_heads if i > 0 else 1
            num_heads = num_heads if i < num_layers - 1 else 1
            out_channels = num_heads

            self.convs.append(
                GATHAConv(
                    in_hidden, 
                    out_hidden, 
                    K=K, 
                    num_heads=num_heads, 
                    edge_drop=edge_drop, 
                    attn_drop=attn_drop, 
                    use_attn_dst=use_attn_dst,
                    norm=norm,
                    residual=True,
                )
            )

            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_dropout = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_dropout(h)

        for i in range(self.num_layers):
            conv = self.convs[i](graph, h)

            h = conv

            if i < self.num_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)
        return h 
        # return F.log_softmax(h, dim=-1)
   
class DHSEGAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 num_hiddens,
                 num_layers,
                 num_heads,
                 activation,
                 n_struc,
                 n_dist,
                 dropout=0.05,
                 attn_drop=0.01):
        super().__init__()
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_struc = n_struc
        self.n_dist = n_dist 
        if n_struc:
            self.struc_encoder = nn.Linear(n_struc, in_feats) 
        if n_dist:
            self.dis_encoder = nn.Linear(n_dist, in_feats)
        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_hidden = num_heads * num_hiddens if i > 0 else in_feats
            out_hidden = num_hiddens if i < num_layers - 1 else n_classes
            out_channels = num_heads

            self.convs.append(GATConv(in_hidden,
                                      out_hidden,
                                      num_heads=num_heads,
                                      attn_drop=attn_drop))
            self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

        # self.bias_last = Bias(n_classes)
        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feats):
        if feats.size()[1] > self.in_feats+self.n_struc+self.n_dist:
            h0 = torch.cat([feats[:, :self.in_feats],feats[:, self.in_feats+self.n_struc+self.n_dist:]],dim=1)    
        else:
            h0 = feats[:, :self.in_feats]
        h0 = F.layer_norm(normalized_shape=[self.in_feats], input=h0)
        if hasattr(self, "struc_encoder"):
            hs = self.struc_encoder(feats[:, self.in_feats:self.in_feats+self.n_struc]) 
            hs = F.layer_norm(normalized_shape=[self.in_feats], input=hs)
        else:
            hs = torch.Tensor([0]).to(h0.device) 
        if hasattr(self, "dist_encoder"):
            hd = self.dis_encoder(feats[:, self.in_feats+self.n_struc:self.in_feats+self.n_struc+self.n_dist]) 
            hd = F.layer_norm(normalized_shape=[self.in_feats], input=hd)
        else:
            hd = torch.Tensor([0]).to(h0.device)
        h = h0 + hs + hd 
        h = self.dropout0(h)
        for i in range(self.num_layers):
            # for mini-batch training 
            if isinstance(graph, list):
                conv = self.convs[i](graph[i], h)
                linear = self.linear[i](h[:graph[i].num_dst_nodes()]).view(conv.shape)
            else:
                conv = self.convs[i](graph, h)
                linear = self.linear[i](h).view(conv.shape)
            h = conv + linear
            if i < self.num_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)
        
        h = h.mean(1)
        h = self.bias_last(h)
        return h 
        # return F.log_softmax(h, dim=-1)
     

class DHSEAGDN(nn.Module):
    def __init__(
        self, 
        in_feats, 
        n_classes, 
        num_hiddens, 
        num_layers, 
        num_heads, 
        activation, 
        K=3, 
        dropout=0.0, 
        input_drop=0.0, 
        edge_drop=0.0, 
        attn_drop=0.05, 
        use_attn_dst=True,
        norm='both',
        n_struc=None,
        n_dist=None,
        
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_struc = n_struc
        self.n_dist = n_dist 
        if n_struc:
            self.struc_encoder = nn.Linear(n_struc, in_feats) 
        if n_dist:
            self.dist_encoder = nn.Linear(n_dist, in_feats)
        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_hidden = num_heads * num_hiddens if i > 0 else in_feats
            out_hidden = num_hiddens if i < num_layers - 1 else n_classes
            out_channels = num_heads

            self.convs.append(GATHAConv(
                    in_hidden, 
                    out_hidden, 
                    K=K, 
                    num_heads=num_heads, 
                    edge_drop=edge_drop, 
                    attn_drop=attn_drop, 
                    use_attn_dst=use_attn_dst,
                    norm=norm,
                    residual=False,
                ))
            self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

        # self.bias_last = Bias(n_classes)
        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_dropout = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feats):
        if feats.size()[1] > self.in_feats+self.n_struc+self.n_dist:
            h0 = torch.cat([feats[:, :self.in_feats],feats[:, self.in_feats+self.n_struc+self.n_dist:]],dim=1)    
        else:
            h0 = feats[:, :self.in_feats]
        h0 = F.layer_norm(normalized_shape=[self.in_feats], input=h0)
        if hasattr(self, "struc_encoder"):
            hs = self.struc_encoder(feats[:, self.in_feats:self.in_feats+self.n_struc]) 
            hs = F.layer_norm(normalized_shape=[self.in_feats], input=hs)
        else:
            hs = torch.Tensor([0]).to(h0.device) 
        if hasattr(self, "dist_encoder"):
            hd = self.dist_encoder(feats[:, self.in_feats+self.n_struc:self.in_feats+self.n_struc+self.n_dist]) 
            hd = F.layer_norm(normalized_shape=[self.in_feats], input=hd)
        else:
            hd = torch.Tensor([0]).to(h0.device)
        h = h0 + hs + hd 
        h = self.input_dropout(h) 
        for i in range(self.num_layers):
            # for mini-batch training 
            if isinstance(graph, list):
                conv = self.convs[i](graph[i], h)
                linear = self.linear[i](h[:graph[i].num_dst_nodes()]).view(conv.shape)
            else:
                conv = self.convs[i](graph, h)
                linear = self.linear[i](h).view(conv.shape)
            h = conv + linear
            if i < self.num_layers - 1:
                h = h.flatten(1)
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)
        
        h = h.mean(1)
        h = self.bias_last(h)
        return h 
        # return F.log_softmax(h, dim=-1)
            
class LabelPropagation(nn.Module):
    r"""

    Description
    -----------
    Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation
    <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

    where unlabeled data is inferred by labeled data via propagation.

    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
        adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
    """
    def __init__(self, num_layers, alpha, adj='DAD'):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
        self.adj = adj
    
    @torch.no_grad()
    def forward(self, g, labels, mask=None, post_step=lambda y: y.clamp_(0., 1.)):
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)
            
            y = labels
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]
            
            last = (1 - self.alpha) * y
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5 if self.adj == 'DAD' else -1).to(labels.device).unsqueeze(1)

            for _ in range(self.num_layers):
                # Assume the graphs to be undirected
                if self.adj in ['DAD', 'AD']:
                    y = norm * y
                
                g.ndata['h'] = y
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = self.alpha * g.ndata.pop('h')

                if self.adj in ['DAD', 'DA']:
                    y = y * norm
                
                y = post_step(last + y)
            
            return y


class CorrectAndSmooth(nn.Module):
    r"""

    Description
    -----------
    Introduced in `Combining Label Propagation and Simple Models Out-performs Graph Neural Networks
    <https://arxiv.org/abs/2010.13993>`_

    Parameters
    ----------
        num_correction_layers: int
            The number of correct propagations.
        correction_alpha: float
            The coefficient of correction.
        correction_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        num_smoothing_layers: int
            The number of smooth propagations.
        smoothing_alpha: float
            The coefficient of smoothing.
        smoothing_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        autoscale: bool, optional
            If set to True, will automatically determine the scaling factor :math:`\sigma`. Default is True.
        scale: float, optional
            The scaling factor :math:`\sigma`, in case :obj:`autoscale = False`. Default is 1.
    """
    def __init__(self,
                 num_correction_layers,
                 correction_alpha,
                 correction_adj,
                 num_smoothing_layers,
                 smoothing_alpha,
                 smoothing_adj,
                 autoscale=True,
                 scale=1.):
        super(CorrectAndSmooth, self).__init__()
        
        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers,
                                      correction_alpha,
                                      correction_adj)

        self.prop2 = LabelPropagation(num_smoothing_layers,
                                      smoothing_alpha,
                                      smoothing_adj)

    def correct(self, g, y_soft, y_true, mask):
        with g.local_scope():
            assert abs(float(y_soft.sum()) / y_soft.size(0) - 1.0) < 1e-2
            # number of nodes that are masked
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            error = torch.zeros_like(y_soft)
            error[mask] = y_true - y_soft[mask]

            if self.autoscale:
                smoothed_error = self.prop1(g, error, post_step=lambda x: x.clamp_(-1., 1.))
                sigma = error[mask].abs().sum() / numel
                scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
                scale[scale.isinf() | (scale > 1000)] = 1.0

                result = y_soft + scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result
            else:
                def fix_input(x):
                    x[mask] = error[mask]
                    return x
                
                smoothed_error = self.prop1(g, error, post_step=fix_input)

                result = y_soft + self.scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result

    def smooth(self, g, y_soft, y_true, mask):
        with g.local_scope():
            numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(y_soft.dtype)
            
            y_soft[mask] = y_true
            return self.prop2(g, y_soft)
