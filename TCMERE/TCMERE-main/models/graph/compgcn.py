from turtle import forward
import torch
from torch import nn
import dgl
import dgl.function as fn
import numpy as np
from IPython import embed


class CompGCNCov(nn.Module):
    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., opn='corr', num_base=-1,
                 num_rel=None):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.opn = opn

        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel * 2, num_base])
        else:
            self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w[edge_dir.squeeze()]).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w.index_select(0, edge_dir.squeeze())).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # NOTE: first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, x, rel_repr, edge_type, edge_norm):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        try:
            g.edata['norm'] = edge_norm
        except:
            embed()
        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3
        if self.bias is not None:
            x = x + self.bias
        x = self.bn(x)

        return self.act(x), torch.matmul(self.rel, self.w_rel)


class CompGCN(nn.Module):
    # def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
    #              conv_bias=True, gcn_drop=0., opn='mult'):
    def __init__(self, etypes, # list(['T132', 'T133', 'T134', 'T193', ...,'T198', 'T199', 'T202'])
                 h_dim, num_bases, # 50, -1
                 num_hidden_layers=1, # 3
                 dropout=0, # 0.1
                 use_self_loop=False, 
                 opn = 'mult'):
        super(CompGCN, self).__init__()
        self.gcn_drop = dropout
        self.opn = opn

        self.h_dim = h_dim
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        self.etype2id = {et:i for i, et in enumerate(self.rel_names)}
        self.num_rel = len(self.rel_names)

        if num_bases < 0 or num_bases > self.num_rel:
            self.num_bases = self.num_rel # 54
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # self.init_embed = self.get_param([self.num_ent, self.init_dim])  # initial embedding for entities
        if self.num_bases > 0:
            # linear combination of a set of basis vectors
            self.init_rel = self.get_param([self.num_bases, self.h_dim])
        else:
            # independently defining an embedding for each relation
            self.init_rel = self.get_param([self.num_rel * 2, self.h_dim])

        self.layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            num_base = self.num_bases if i==0 else -1
            self.layers.append(CompGCNCov(
                self.h_dim, self.h_dim, act=torch.tanh, bias=True, drop_rate=self.gcn_drop, \
                    opn=self.opn, num_base=num_base, num_rel=self.num_rel
            ))
 

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def to_homograph(self, hg):
        lst, src_l, tgt_l = [], [], []
        for etype in hg.etypes:
            src, tgt = hg.edges(etype=etype)
            src_l.append(src)
            tgt_l.append(tgt)
            idx = self.etype2id[etype]
            lst.extend([idx]*len(src))

        src, tgt = torch.hstack(src_l), torch.hstack(tgt_l)
        g = dgl.graph((src, tgt)).to(hg.device)
        edge_type = torch.tensor(lst).to(hg.device)
        return g, edge_type
    
    def calc_edge_norm(self, g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float() # torch.tensor() [45]
        norm = in_deg ** -0.5
        norm[torch.isinf(norm)] = 0
        g.ndata['xxx'] = norm
        g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        edge_norm = g.edata.pop('xxx').squeeze()
        return edge_norm
        
    def forward(self, hg, dic):
        kk, x = list(dic.items())[0]
        g, edge_type = self.to_homograph(hg)
        edge_norm = self.calc_edge_norm(g)

        r = self.init_rel
        for layer in self.layers:
            x, r = layer(g, x, r, edge_type, edge_norm)
        return {kk: x}
    

if __name__ == '__main__':
    # device = 'cuda'
    # hg, _ = dgl.load_graphs("../../ekg_graph.bin")
    # hg = hg[0].to(device)
    # print(hg)

    # x = torch.rand((45, 50)).to(device)
    # etypes = ['T132']*54
    etypes = ['T132', 'T133']
    # etypes = [str(i) for i in range(54)]
    # model = CompGCN(etypes, 50, -1, 3, 0.1).to(device)

    # out = model(hg, x)
    # print(out.shape)

    src, tgt = [0, 1, 0, 3, 2], [1, 3, 3, 4, 4]
    g = dgl.graph((src, tgt))
    # g.add_nodes(5)
    # g.add_edges(src, tgt)  # src -> tgt
    # g.add_edges(tgt, src)  # tgt -> src
    edge_type = torch.tensor([0, 0, 0, 1, 1] + [2, 2, 2, 3, 3])
    x = torch.rand((g.number_of_nodes(), 50))

    embed()

    model = CompGCN(2, 50, -1, 3, 0.1)

    out = model(g, x)
    print(out.shape)