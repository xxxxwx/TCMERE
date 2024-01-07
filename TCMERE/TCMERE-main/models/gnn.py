import torch
import torch.nn as nn
from models.base import FFNNModule
# from models.external_knowledge import BiGCN
from models.graph.compgcn import CompGCNCov
from constants import *
from utils import tolist
import dgl
from models.graph.gcn import GraphConvolution
from IPython import embed


class BiGCNLayer(nn.Module):
    def __init__(self, etypes, configs):
        super(BiGCNLayer, self).__init__()
        self.etypes = etypes
        self.num_rels = len(etypes)
        self.configs = configs
        self.hid_size = configs['span_emb_size']

        gcn2p_fw, gcn2p_bw = [], []
        for _ in range(self.num_rels):
            gcn2p_fw.append(GraphConvolution(self.hid_size, self.hid_size // 2))
            gcn2p_bw.append(GraphConvolution(self.hid_size, self.hid_size // 2))
        self.gcn2p_fw = nn.ModuleList(gcn2p_fw)
        self.gcn2p_bw = nn.ModuleList(gcn2p_bw)

        self.dropout = nn.Dropout(configs['ieg_bignn_dropout'])
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.hid_size, self.hid_size)

    def forward(self, inps, fw_adjs, bw_adjs):
        # No self-loops in fw_adjs or bw_adjs
        num_rels = self.num_rels
        assert(len(fw_adjs) == num_rels)
        assert(len(bw_adjs) == num_rels)

        outs = []
        for i in range(num_rels):
            fw_outs = self.gcn2p_fw[i](inps, fw_adjs[i])
            bw_outs = self.gcn2p_bw[i](inps, bw_adjs[i])
            outs.append(self.dropout(torch.cat([bw_outs, fw_outs], dim=-1)))
        outs = torch.cat([o.unsqueeze(0) for o in outs], dim=0)

        feats = self.linear1(self.relu(torch.sum(outs, dim=0)))
        feats += inps # Residual connection
        return feats


class BiGCN(nn.Module):
    def __init__(self, etypes, configs):
        super(BiGCN, self).__init__()
        self.etypes = etypes
        self.configs = configs
        self.num_hidden_layers = configs['ieg_bignn_hidden_layers']

        bigcn_layers = []
        for _ in range(self.num_hidden_layers):
            bigcn_layers.append(BiGCNLayer(etypes, configs))
        self.bigcn_layers = nn.ModuleList(bigcn_layers)

    def forward(self, embs, fw_adjs, bw_adjs):
        out = embs
        for i in range(self.num_hidden_layers):
            out = self.bigcn_layers[i](out, fw_adjs, bw_adjs)
        return out



class CompGCN(nn.Module):
    def __init__(self, etypes, configs, device, num_base=-1, opn='mult'):
        super(CompGCN, self).__init__()
        self.etypes = etypes
        self.num_rels = len(etypes)
        self.configs = configs
        self.num_hidden_layers = configs['ieg_bignn_hidden_layers']

        self.num_base = num_base
        self.h_dim = self.configs['span_emb_size']
        self.dropout = self.configs['ieg_bignn_dropout']
        self.device = device

        if self.num_base > 0:
            # linear combination of a set of basis vectors
            self.init_rel = self.get_param([self.num_base, self.h_dim])
        else:
            # independently defining an embedding for each relation
            self.init_rel = self.get_param([self.num_rels * 2, self.h_dim])

        self.layers = nn.ModuleList()
        for i in range(self.num_hidden_layers):
            num_base = self.num_base if i==0 else -1
            self.layers.append(CompGCNCov(
                self.h_dim, self.h_dim, act=torch.tanh, bias=True, drop_rate=self.dropout, \
                    opn=opn, num_base=num_base, num_rel=self.num_rels
            ))


    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param


    def calc_edge_norm(self, g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float() # torch.tensor() [45]
        norm = in_deg ** -0.5
        norm[torch.isinf(norm)] = 0
        g.ndata['xxx'] = norm
        g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        edge_norm = g.edata.pop('xxx')
        return edge_norm


    def build_graph(self, fw_adjs):
        lst, src_l, tgt_l = [], [], []
        for idx, fw_adj in enumerate(fw_adjs):
            adj = (fw_adj>0.5).to(torch.int)
            edge_index = torch.nonzero(adj)
            src, tgt = edge_index[:, 0], edge_index[:, 1]

            src_l.append(src)
            tgt_l.append(tgt)
            lst.extend([idx]*len(src))
        
        src, tgt = torch.hstack(src_l), torch.hstack(tgt_l)
        g = dgl.graph((src, tgt)).to(self.device)
        edge_type = torch.tensor(lst).to(self.device)
        return g, edge_type


    def forward(self, x, fw_adjs, bw_adjs=None):
        # embed()
        g, edge_type = self.build_graph(fw_adjs)
        edge_norm = self.calc_edge_norm(g)

        # print('num_nodes:', len(x))
        if g.number_of_nodes() == len(x):
            # try:
            r = self.init_rel
            for layer in self.layers:
                x, r = layer(g, x, r, edge_type, edge_norm)
            # except:
            #     embed()
        return x


class Gnn(nn.Module):
    def __init__(self, configs, device, final_head=True, typed=True):
        super(Gnn, self).__init__()
        self.device = device
        self.span_emb_size = configs['span_emb_size']  # 512
        self.pair_embs_size = 3 * self.span_emb_size
        self.nb_relation_types = len(configs['relation_types']) if typed else 2
        self.final_head = final_head

        # Relation Extractor
        relation_hidden_sizes = [configs['mention_linker_ffnn_size']] * configs['mention_linker_ffnn_depth']
        relation_scorer_input_size = self.pair_embs_size
        self.relation_scorer = FFNNModule(input_size = relation_scorer_input_size,
                                                hidden_sizes = relation_hidden_sizes,
                                                output_size = self.nb_relation_types,
                                                dropout = configs['dropout_rate'])
        # GCNs for prior IE graph
        #if configs['dataset'] == ADE: nb_ieg_etypes = len(ADE_RELATION_TYPES)
        if configs['dataset'] == BIORELEX: nb_ieg_etypes = len(BIORELEX_RELATION_TYPES)
        self.ieg_etypes = list(range(nb_ieg_etypes))

        self.gnn_mode = configs['gnn_mode'].lower()
        if self.gnn_mode == 'compgcn':
            self.gnn = CompGCN(self.ieg_etypes, configs, device)
        else:
            self.gnn = BiGCN(self.ieg_etypes, configs)


    def forward(self, candidate_embs): # [n, h=512]
        # Compute pair_embs
        pair_embs = get_pair_embs(candidate_embs) # [n, n, 3*h]

        # Compute pair_relation_scores and pair_relation_loss
        pair_relation_scores = self.relation_scorer(pair_embs) # [n, n, 4]
        if len(pair_relation_scores.size()) <= 1: # False
            pair_relation_scores = pair_relation_scores.view(1, 1, self.nb_relation_types)
        
        # Compute probs
        relation_probs = torch.softmax(pair_relation_scores, dim=-1) # [n, n, 4]

        # Process prior IE predictions
        fw_adjs, bw_adjs = self.adjs_from_preds(relation_probs) # list(4*[tensor([n, n])])
        ieg_out_h = self.gnn(candidate_embs, fw_adjs, bw_adjs) # [n, h]

        return ieg_out_h


    def adjs_from_preds(self, relation_probs):
        relation_probs = relation_probs.clone().detach()
        fw_adjs, bw_adjs = [], []
        # nb_nodes = relation_probs.size()[0]
        for ix in range(len(self.ieg_etypes)):
            A = relation_probs[:,:,ix]
            # Fill diagonal with zero
            A.fill_diagonal_(0)
            # Update fw_adjs and bw_adjs
            fw_adjs.append(A.to(self.device))
            bw_adjs.append(A.T.to(self.device))
        return fw_adjs, bw_adjs


def get_pair_embs(candidate_embs):
    n, d = candidate_embs.size()
    features_list = []

    # Compute diff_embs and prod_embs
    src_embs = candidate_embs.view(1, n, d).repeat([n, 1, 1])
    target_embs = candidate_embs.view(n, 1, d).repeat([1, n, 1])
    prod_embds = src_embs * target_embs

    # Update features_list
    features_list.append(src_embs)
    features_list.append(target_embs)
    features_list.append(prod_embds)

    # Concatenation
    pair_embs = torch.cat(features_list, 2)

    return pair_embs
    

