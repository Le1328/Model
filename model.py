import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import clones, get_mean, device
from mprnn.gcrnn import GCGRUCell


class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dim, h, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        assert input_dim % h == 0

        self.d_k = input_dim // h
        self.h = h
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)

    def forward(self, query, key):
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]

        attn = self.attention(query.squeeze(2), key.squeeze(2))

        return attn

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1,0,2), key.permute(1,2,0)) \
                 / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1,2,0)).squeeze(2)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


class MPGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 support_len=1,
                 n_nodes=None,
                 layer_norm=False,
                 autoencoder_mode=False):
        super(MPGRU, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.ff_size = int(ff_size) if ff_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size

        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(GCGRUCell(d_in=rnn_input_size if i == 0 else self.hidden_size,
                                        num_units=self.hidden_size, support_len=support_len, order=kernel_size))
            if layer_norm:
                self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size))
            else:
                self.norms.append(nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.pred_readout = nn.Linear(self.hidden_size, self.input_size)

        self.w_dg_x = nn.Linear(input_size, input_size)
        self.w_dg_h = nn.Linear(input_size, hidden_size)

        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter('h0', None)

        self.autoencoder_mode = autoencoder_mode

    def init_hidden_states(self, n_nodes):
        h0 = []
        for l in range(self.n_layers):
            std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, delta, mask):
        batch_size = x.size(0)
        time_step = x.size(1)
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        *_, steps = x.size()

        h = self.get_h0(x)

        gamma_h = torch.exp(
            -torch.max(torch.zeros(batch_size, time_step, self.hidden_size).to(device), self.w_dg_h(delta)))

        predictions, states = [], []
        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]

            x_s_hat = self.pred_readout(h_s.permute(0, 2, 1)).permute(0, 2, 1)

            predictions.append(x_s_hat)
            states.append(torch.stack(h, dim=0))

            x_s = torch.where(m_s, x_s, x_s_hat)

            h = [torch.mul(gamma_h[:, step, :].T, torch.stack(h, dim=0).squeeze()).unsqueeze(0)]

            inputs = [x_s, m_s]
            inputs = torch.cat(inputs, dim=1)

            h = self.update_state(inputs, h, adj)

        if self.autoencoder_mode:
            states = states[1:] + [torch.stack(h, dim=0)]

        predictions = torch.stack(predictions, dim=-1)
        states = torch.stack(states, dim=-1)

        predictions = predictions.squeeze().permute(1, 2, 0)
        states = states.squeeze().permute(1, 2, 0)

        return predictions, states


class GraphFusion(nn.Module):
    def __init__(self, proj_dim):
        super(GraphFusion, self).__init__()
        self.proj_ts = nn.Linear(256, proj_dim)
        self.proj_diag = nn.Linear(256, proj_dim)
        self.proj_demo = nn.Linear(256, proj_dim)

        self.proj_all = nn.Linear(256 * 3, proj_dim)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, adj_ts, adj_diag, adj_demo):
        if adj_ts.shape[1] != 256:
            adj_ts = torch.cat((adj_ts, torch.zeros(adj_ts.shape[0], 256 - adj_ts.shape[1]).to(device)), 1)
            adj_diag = torch.cat((adj_diag, torch.zeros(adj_diag.shape[0], 256 - adj_diag.shape[1]).to(device)), 1)
            adj_demo = torch.cat((adj_demo, torch.zeros(adj_demo.shape[0], 256 - adj_demo.shape[1]).to(device)), 1)

        adj_all = torch.cat((adj_ts, adj_diag, adj_demo), 1)
        q_all = self.proj_all(adj_all)
        q_ts = self.proj_ts(adj_ts)
        q_diag = self.proj_diag(adj_diag)
        q_demo = self.proj_demo(adj_demo)

        d_k = q_all.size(-1)
        scores_ts = torch.matmul(q_all, q_ts.T) / math.sqrt(d_k)
        scores_diag = torch.matmul(q_all, q_diag.T) / math.sqrt(d_k)
        scores_demo = torch.matmul(q_all, q_demo.T) / math.sqrt(d_k)
        scores = torch.stack([scores_ts, scores_diag, scores_demo])
        att = self.softmax(scores)
        att_ts = att[0]
        att_diag = att[1]
        att_demo = att[2]

        return att_ts, att_diag, att_demo


class Model(nn.Module):
    def __init__(self, x_dim, diag_dim, demo_dim, h_ts, h_diag, h_demo, \
                 proj_dim, hidden_dim, drop_prob, task, n_nodes=None):
        super(Model, self).__init__()
        self.task = task
        self.gru = nn.GRU(input_size=x_dim, hidden_size=x_dim, batch_first=True, bidirectional=False)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, x_dim), requires_grad=True).to(device)

        self.MHA_ts = MultiHeadedAttention(x_dim, h_ts, drop_prob)
        self.MHA_diag = MultiHeadedAttention(diag_dim, h_diag, drop_prob)
        self.MHA_demo = MultiHeadedAttention(demo_dim, h_demo, drop_prob)
        self.gfusion = GraphFusion(proj_dim)
        self.mpgru = MPGRU(input_size=x_dim + diag_dim + demo_dim, hidden_size=hidden_dim, ff_size=None, n_layers=1,
                           dropout=drop_prob, kernel_size=3,
                           support_len=1, n_nodes=n_nodes, layer_norm=False, autoencoder_mode=True).to(device)
        self.pre = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, diag, demo, delta, sorted_lengths):
        mask = torch.where(x != -1, torch.ones(x.shape).to(device), torch.zeros(x.shape).to(device))

        diag_tmp = diag.unsqueeze(1).expand(x.shape[0], x.shape[1], diag.shape[1]).to(device)
        demo_tmp = demo.unsqueeze(1).expand(x.shape[0], x.shape[1], demo.shape[1]).to(device)
        x_tmp = torch.cat((x, diag_tmp, demo_tmp), 2)
        x_tmp = x_tmp.unsqueeze(0).permute(0, 3, 1, 2)
        mask2 = torch.where(x_tmp != -1, torch.ones(x_tmp.shape, dtype=torch.uint8, device=device),
                            torch.zeros(x_tmp.shape, dtype=torch.uint8, device=device))

        t_mean, s_mean = get_mean(x, mask)
        x_mean = 0.5 * t_mean + (1 - 0.5) * s_mean
        x = torch.where(x == -1, x_mean, x)

        batch_size = x.size(0)
        h_0_contig = self.h_0.expand(1, batch_size, self.gru.hidden_size).contiguous()
        gru_out, _ = self.gru(x, h_0_contig)

        gru_list = []
        for i in range(gru_out.shape[0]):
            idx = sorted_lengths[i] - 1
            gru_list.append(gru_out[i, idx, :])
        gru_ = torch.stack(gru_list)

        adj_ts = self.MHA_ts(gru_, gru_)
        adj_diag = self.MHA_diag(diag, diag)
        adj_demo = self.MHA_demo(demo, demo)
        att_ts, att_diag, att_demo = self.gfusion(adj_ts, adj_diag, adj_demo)
        adj = att_ts * adj_ts + att_diag * adj_diag + att_demo * adj_demo

        diag = diag.unsqueeze(1).expand(x.shape[0], x.shape[1], diag.shape[1]).to(device)
        demo = demo.unsqueeze(1).expand(x.shape[0], x.shape[1], demo.shape[1]).to(device)
        input = torch.cat((gru_out, diag, demo), 2)
        x_hat, hidden_states = self.mpgru(input, adj, delta, mask2)

        if self.task == 'Imputation':
            return x_hat

        h_list = []
        for i in range(hidden_states.shape[0]):
            idx = sorted_lengths[i] - 1
            h_list.append(hidden_states[i, idx, :])
        h_out = torch.stack(h_list)

        output = self.pre(h_out)

        return output

