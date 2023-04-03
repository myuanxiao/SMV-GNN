from models.gcl import E_GCL, unsorted_segment_sum
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)
        return h


class Prediction(nn.Module):
    def __init__(self, in_node_nf, in_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):

        super(Prediction, self).__init__()
        self.in_nf = in_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.node_dec = nn.Sequential(nn.Linear(self.in_nf, self.hidden_nf),
                                      act_fn)

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h, node_mask, n_nodes):

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


class PredictionKL(nn.Module):
    def __init__(self, in_node_nf, in_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):

        super(PredictionKL, self).__init__()
        self.in_nf = in_nf
        self.hidden_nf = hidden_nf
        self.device = device

        self.graph_dec = nn.Sequential(nn.Linear(self.in_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h):

        pred = self.graph_dec(h)
        return pred.squeeze(1)

class Readout(nn.Module):
    def __init__(self, in_node_nf, in_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):

        super(Readout, self).__init__()
        self.in_nf = in_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.node_dec = nn.Sequential(nn.Linear(self.in_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h, node_mask, n_nodes):

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


# raw h:   torch.Size([2400, 128])
# node_de: torch.Size([2400, 128])
# review, sum dim=1 torch.Size([96, 128])
# graph_dec torch.Size([96, 1])

class SSL_3d(nn.Module):
    def __init__(self, hidden_nf, device='cpu', act_fn=nn.SiLU()):

        super(SSL_3d, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn)

        self.pre_ra = nn.Sequential(nn.Linear(self.hidden_nf, 3),
                                      torch.nn.Softmax(dim=2))
        self.to(self.device)

    def forward(self, h, node_mask, n_nodes):

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = self.pre_ra(h)

        return h



class SSL_2d(nn.Module):
    def __init__(self, hidden_nf, device='cpu', act_fn=nn.SiLU()):

        super(SSL_2d, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn)

        self.pre_di = nn.Sequential(nn.Linear(self.hidden_nf, 1))

        self.to(self.device)

    def forward(self, h, node_mask, n_nodes):

        h = self.node_dec(h)
        N = h.size()[1] # number of nodes
        h = self.pre_di(h.repeat_interleave(N, dim=1) - h.repeat(1, N, 1))

        return torch.reshape(h, (h.size()[0], N, N)).long()


class PredictionTox21(nn.Module):
    def __init__(self, in_node_nf, in_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):

        super(PredictionTox21, self).__init__()
        self.in_nf = in_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.node_dec = nn.Sequential(nn.Linear(self.in_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 2))
        self.to(self.device)

    def forward(self, h, node_mask, n_nodes):

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        # print('dec h 1', h.shape)
        h = torch.sum(h, dim=1)
        # print('dec h 2', h.shape)
        out_h = h
        pred = self.graph_dec(h)
        return pred.squeeze(1), out_h

class PredictionKL_C(nn.Module):
    def __init__(self, in_node_nf, in_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):

        super(PredictionKL_C, self).__init__()
        self.in_nf = in_nf
        self.hidden_nf = hidden_nf
        self.device = device

        self.graph_dec = nn.Sequential(nn.Linear(self.in_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 2))
        self.to(self.device)

    def forward(self, h):

        pred = self.graph_dec(h)
        return pred.squeeze(1)

class PredictionQm8(nn.Module):
    def __init__(self, in_node_nf, in_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):

        super(PredictionQm8, self).__init__()
        self.in_nf = in_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.node_dec = nn.Sequential(nn.Linear(self.in_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h, node_mask, n_nodes):

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_nf, edges_in_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.edges_in_nf = edges_in_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        #self.add_module("gcl_0", GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=self.edges_in_nf, act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)


    def forward(self, nodes, edges, edge_mask, edge_attr=None):
        h = self.embedding(nodes)
        #h, _ = self._modules["gcl_0"](h, edges, edge_attr=edge_attr)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_mask, edge_attr=edge_attr)
        #return h
        return h

class MergeGnn(nn.Module):
    def __init__(self, input_dim, hidden_nf, edges_in_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=0, recurrent=False):
        super(MergeGnn, self).__init__()
        self.hidden_nf = hidden_nf
        self.edges_in_nf = edges_in_nf
        self.device = device
        self.n_layers = n_layers
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data[0]  = 0.5
        ### Encoder
        #self.add_module("gcl_0", GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=self.edges_in_nf, act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)


    def forward(self, nodes, edges, edges_2d, edges_3d, edge_attr=None):
        h = self.embedding(nodes)
        edge_mask = self.alpha * edges_2d + (1-self.alpha) * edges_3d
        print("Alpha:",self.alpha.item())
        #h, _ = self._modules["gcl_0"](h, edges, edge_attr=edge_attr)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_mask, edge_attr=edge_attr)
        #return h
        return h




class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()


    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_mask, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        edge_feat = edge_feat * edge_mask
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat



class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        # edges_in_nf = 18
        print("input_edge_nf + edges_in_nf", input_edge_nf + edges_in_nf)
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        #if recurrent:
            #self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            #out = self.gru(out, h)
        return out
