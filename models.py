import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
import igraph
import pdb

# This file implements several VAE models for DAGs, including SVAE, GraphRNN, DVAE, GCN etc.

'''
    String Variational Autoencoder (S-VAE). Treat DAGs as sequences of node descriptors. 
    A node descriptor is the concatenation of the node type's one-hot encoding and its 
    bit connections from other nodes. Nodes in a sequence are in a topological order.
'''
class SVAE(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False):
        super(SVAE, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.xs = (nvt + max_n-1)  # size of input x for GRU,
                                   # [one_hot(vertex_type), bit(connections)]
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # graph state size
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.device = None

        # 0. encoding-related
        self.grue = nn.GRU(self.xs, hs, batch_first=True, bidirectional=self.bidir)  # encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar
            
        # 1. decoding-related
        self.grud = nn.GRU(hs, hs, batch_first=True)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, self.nvt), 
                )
        self.add_edges = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, self.max_n - 1), 
                )

        # 2. bidir-related, to unify sizes
        if self.bidir:
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.hs * 2, self.hs), 
                    )
            self.hv_unify = nn.Sequential(
                    nn.Linear(self.hs * 2, self.hs), 
                    )

        # 3. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) == list:
            if idx == []:
                return None
            idx = torch.LongTensor([[i] for i in idx])
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _collate_fn(self, G):
        # create mini_batch of tensors from list G by padding
        # each graph g is a 1 * (n_vertex - 1) * (n_types + n_vertex-1) tensor
        # pad all to 1 * (max_n - 1) * (n_types + max_n-1) tensors
        G_new = []
        for g in G:
            if g.shape[1] < self.max_n - 1:
                padding = torch.zeros(1, self.max_n-1-g.shape[1], g.shape[2]).to(self.get_device())
                padding[0, :, self.START_TYPE] = 1  # use start type's bit to indicate padding 
                                                    # nodes (since start types are never predicted)
                g = torch.cat([g, padding], 1)
            if g.shape[2] < self.xs:
                padding = torch.zeros(1, g.shape[1], self.xs-g.shape[2]).to(self.get_device())
                g = torch.cat([g, padding], 2)  # pad zeros to indicate no connections to padding 
                                                # nodes
            G_new.append(g)
        return torch.cat(G_new, 0)

    def encode(self, G):
        # G: [batch_size * max_n-1 * xs]
        _, Hn = self.grue(G)
        Hg = Hn.view(Hn.shape[1], -1)   # Hn's second dimension is "batch"
        if self.bidir:
            Hg = self.hg_unify(Hg)
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        #return mu
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _decode(self, z):
        H0 = self.relu(self.fc3(z))
        H_in = H0.unsqueeze(1).expand(-1, self.max_n - 1, -1)
        H_out, _ = self.grud(H_in)
        type_scores = self.add_vertex(H_out)  # batch * max_n-1 * nvt
        edge_scores = self.sigmoid(self.add_edges(H_out))  # batch * max_n-1 * max_n-1
        return type_scores, edge_scores

    def decode(self, z):
        type_scores, edge_scores = self._decode(z)
        return self.construct_igraph(type_scores, edge_scores)

    def loss(self, mu, logvar, G_true, beta=0.005):
        # G_true: [batch_size * max_n-1 * xs]
        z = self.reparameterize(mu, logvar)
        type_scores, edge_scores = self._decode(z)
        res = 0
        _, true_types = torch.max(G_true[:, :, :self.nvt], 2)
        res += F.cross_entropy(type_scores.transpose(1, 2), true_types, reduction='sum')
        true_edges = G_true[:, :, self.nvt:]
        res += F.binary_cross_entropy(edge_scores, true_edges, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def construct_igraph(self, type_scores, edge_scores, stochastic=True):
        # construct igraphs from node type and edge scores
        # note that when stochastic=True, type_scores should be raw scores before softmax, 
        # and edge_scores should probabilities between [0, 1] (after sigmoid)
        assert(type_scores.shape[:2] == edge_scores.shape[:2])
        if stochastic:
            type_probs = F.softmax(type_scores, 2).cpu().detach().numpy()
        G = []
        for gi in range(len(type_scores)):
            g = igraph.Graph(directed=True)
            g.add_vertex(type=self.START_TYPE)
            for vj in range(1, self.max_n):
                if vj == self.max_n - 1:
                    new_type = self.END_TYPE
                else:
                    if stochastic:
                        new_type = np.random.choice(range(self.nvt), p=type_probs[gi][vj-1])
                    else:
                        new_type = torch.argmax(type_scores[gi][vj-1], 0).item()
                g.add_vertex(type=new_type)
                if new_type == self.END_TYPE:  
                    end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                        if v.index != g.vcount()-1])
                    for v in end_vertices:
                        g.add_edge(v, vj)
                    break
                else:
                    for ek in range(vj):
                        ek_score = edge_scores[gi][vj-1][ek].item()
                        if stochastic:
                            if np.random.random_sample() < ek_score:
                                g.add_edge(ek, vj)
                        else:
                            if ek_score > 0.5:
                                g.add_edge(ek, vj)
            G.append(g)
        return G

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, _, _ = self.loss(mu, logvar, G)
        return loss

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G


'''
    One-shot version of S-VAE. Encode/decode the entire matrix in one shot.
'''
class SVAE_oneshot(SVAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=1002, nz=112, bidirectional=False):
        super(SVAE_oneshot, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.encoder_nn = nn.Sequential(
                nn.Linear((max_n-1) * self.xs, 2 * (max_n-1) * self.xs), 
                nn.ReLU(), 
                nn.Linear(2 * (max_n-1) * self.xs, self.gs), 
                )
        self.decoder_nn = nn.Sequential(
                nn.Linear(hs, 2 * hs), 
                nn.ReLU(), 
                nn.Linear(2 * hs, (max_n-1) * self.xs), 
                )

    def encode(self, G):
        # G: [batch_size * max_n-1 * xs]
        Hg = self.relu(self.encoder_nn(G.view(len(G), -1)))
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def _decode(self, z):
        H0 = self.relu(self.fc3(z))
        scores = self.decoder_nn(H0).view(len(z), self.max_n-1, -1)
        type_scores = scores[:, :, :self.nvt]  # batch * max_n-1 * nvt
        edge_scores = self.sigmoid(scores[:, :, self.nvt:])  # batch * max_n-1 * max_n-1
        return type_scores, edge_scores


'''
    S-VAE with GraphRNN as the decoder. Encode/decode the entries of each adjacency row using 
    another GRU. Use a topological order (instead of BFS) to generate nodes.
    Use teacher forcing during training (use ground truth nodes/edges in each step).
'''
class SVAE_GraphRNN(SVAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False):
        super(SVAE_GraphRNN, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.num_dirs = 2 if self.bidir else 1
        self.num_layers = 1
        self.grud = nn.GRU(self.xs, hs, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidir)  # encoder GRU (graph level)
        self.grud_edge = nn.GRU(1, hs, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidir)  # encoder GRU (edge level)

        self.add_edge = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, 1), 
                )

    def _decode(self, z):
        H0 = self.relu(self.fc3(z))  # batch * hs
        H0_graph = H0.unsqueeze(0).expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
        input_graph_level = self._get_zeros(len(z), self.xs).unsqueeze(1)  # batch * 1 * xs
        type_scores, edge_scores = [], []
        for vi in range(self.max_n-1):
            output_graph_level, _ = self.grud(input_graph_level, H0_graph)  # batch * 1 * (hs*num_dirs)
            if self.bidir:
                output_graph_level = self.hg_unify(output_graph_level)  # batch * 1 * hs
            H0_graph = output_graph_level.permute(1, 0, 2)  # 1 * batch * hs
            H0_graph = H0_graph.expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
            type_score = self.add_vertex(output_graph_level)  # batch * 1 * nvt
            type_prob = F.softmax(type_score, 2).squeeze(1)  # batch * nvt
            new_type = torch.multinomial(type_prob, 1)  # batch * 1
            type_score = self._one_hot(new_type.reshape(-1).tolist(), self.nvt).unsqueeze(1)  # batch * 1 * nvt
            type_scores.append(type_score)
            H0_edge = output_graph_level.permute(1, 0, 2)  # 1 * batch * hs
            H0_edge = H0_edge.expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
            input_edge_level = self._get_zeros(len(z), 1).unsqueeze(1)  # batch * 1 * 1
            edge_score = []
            for ej in range(self.max_n-1):
                output_edge_level, _ = self.grud_edge(input_edge_level, H0_edge)  # batch * 1 * (hs*num_dirs)
                if self.bidir:
                    output_edge_level = self.hv_unify(output_edge_level)  # batch * 1 * hs
                H0_edge = output_edge_level.permute(1, 0, 2)
                H0_edge = H0_edge.expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
                edge_score_j = self.sigmoid(self.add_edge(output_edge_level)).detach()  # batch * 1 * 1
                # sample edge j of node i
                edge_score_j = np.random.random_sample((len(z), 1, 1)) < edge_score_j
                edge_score_j = edge_score_j.type(torch.FloatTensor).to(self.get_device())
                edge_score.append(edge_score_j)
                input_edge_level = edge_score_j
            edge_score = torch.cat(edge_score, 2)  # batch * 1 * max_n-1
            edge_scores.append(edge_score)
            input_graph_level = torch.cat([type_score, edge_score], 2)  # batch * 1 * xs

        type_scores = torch.cat(type_scores, 1)  # batch * max_n-1 * nvt
        edge_scores = torch.cat(edge_scores, 1)  # batch * max_n-1 * max_n-1

        return type_scores, edge_scores

    def decode(self, z):
        type_scores, edge_scores = self._decode(z)
        return self.construct_igraph(type_scores, edge_scores, stochastic=False)

    def loss(self, mu, logvar, G_true, beta=0.005):
        # G_true: [batch_size * max_n-1 * xs]
        # use teacher forcing to train (feed groundtruth at each step instead of prediction)
        z = self.reparameterize(mu, logvar)
        H0 = self.relu(self.fc3(z))  # batch * hs
        Input_graph_level = G_true[:, :-1, :].contiguous()
        Input_graph_level = torch.cat([self._get_zeros(len(z), self.xs).unsqueeze(1), Input_graph_level], 1)  # pad initial zeros
        H0_graph = H0.unsqueeze(0).expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
        Output_graph_level, _ = self.grud(Input_graph_level, H0_graph)  # batch * max_n-1 * (hs*num_dirs)
        if self.bidir:
            Output_graph_level = self.hg_unify(Output_graph_level)  # batch * max_n-1 * hs
        type_scores = self.add_vertex(Output_graph_level)  # batch * max_n-1 * nvt

        # merge node dimension with batch dimension as the "new" batch dimension for parallel computing
        H0_edge = Output_graph_level
        H0_edge = H0_edge.reshape(1, -1, self.hs)  # 1 * (batch * max_n-1) * hs
        H0_edge = H0_edge.expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
        Input_edge_level = G_true[:, :, self.nvt:-1]
        Input_edge_level = torch.cat([self._get_zeros(len(z), self.max_n-1).unsqueeze(2), Input_edge_level], 2)  # pad initial zeros
        Input_edge_level = Input_edge_level.reshape(-1, self.max_n-1, 1).contiguous()  # (batch * max_n-1) * max_n-1 * 1
        Output_edge_level, _ = self.grud_edge(Input_edge_level, H0_edge)  # (batch * max_n-1) * max_n-1 * (hs * num_dirs)
        if self.bidir:
            Output_edge_level = self.hv_unify(Output_edge_level)  # (batch * max_n-1) * max_n-1 * hs
        edge_scores = self.sigmoid(self.add_edge(Output_edge_level))  # (batch * max_n-1) * max_n-1 * 1
        edge_scores = edge_scores.reshape(-1, self.max_n-1, self.max_n-1)

        res = 0
        _, true_types = torch.max(G_true[:, :, :self.nvt], 2)
        res += F.cross_entropy(type_scores.transpose(1, 2), true_types, reduction='sum')
        true_edges = G_true[:, :, self.nvt:]
        res += F.binary_cross_entropy(edge_scores, true_edges, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld


'''
    GraphRNN decoder using a random BFS order
'''
from collections import deque
class SVAE_GraphRNN_BFS(SVAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False):
        super(SVAE_GraphRNN_BFS, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.num_dirs = 2 if self.bidir else 1
        self.num_layers = 1
        self.xs = (nvt + max_n)  # size of input x for GRU,
                                 # [one_hot(vertex_type), bit(connections)]
        self.grue = nn.GRU(self.xs, hs, batch_first=True, bidirectional=self.bidir)  # encoder GRU
        self.grud = nn.GRU(self.xs, hs, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidir)  # decoder GRU (graph level)
        self.grud_edge = nn.GRU(1, hs, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidir)  # decoder GRU (edge level)

        self.add_edge = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, 1), 
                )

    def bfs(self, adj, feat):
        n = len(adj)
        queue = deque([random.randint(0, n-1)])
        visited = set()
        order = []
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            order.append(cur)
            visited.add(cur)
            successors = adj[cur].nonzero().flatten().tolist()
            predecessors = adj[:, cur].nonzero().flatten().tolist()
            neis = set(successors + predecessors)
            neis = neis - visited
            for x in neis:
                queue.append(x)
        return adj[order, :][:, order], feat[order]

    def G_to_adjfeat(self, G):
        # convert SVAE's G tensor to adjacency matrix and node features
        assert(G.shape[0]==1)
        G = G[0]
        pad = torch.zeros(1, self.nvt).to(G.device)
        pad[:, 0] = 1
        input_features = torch.cat([pad, G[:, :self.nvt]], 0)  # add the start node
        pad2 = torch.zeros(self.max_n-1, 1).to(G.device)
        adj = torch.cat([pad2, G[:, self.nvt:].permute(1, 0)], 1)
        pad3 = torch.zeros(1, self.max_n).to(G.device)
        adj = torch.cat([adj, pad3], 0)
        return adj, input_features

    def adjfeat_to_G(self, adj, feat):
        # the new G tensor contains starting node as well as connections of last node
        adj = adj.permute(1, 0)
        return torch.cat([feat, adj], 1).unsqueeze(0)

    def _collate_fn(self, G):
        # create mini_batch of tensors from list G by padding
        # each graph g is a 1 * (n_vertex - 1) * (n_types + n_vertex-1) tensor
        # first transform each g into its node features and adj matrix
        # then each graph is permuted by a bfs node order
        # pad all to 1 * (max_n) * (n_types + max_n) tensors
        G_new = []
        for g in G:
            # apply a bfs ordering to nodes
            adj, feat = self.G_to_adjfeat(g)
            g = self.adjfeat_to_G(*self.bfs(adj, feat))  # 1 * n_vertex * (n_types + n_vertex)
            if g.shape[1] < self.max_n:
                padding = torch.zeros(1, self.max_n-g.shape[1], g.shape[2]).to(self.get_device())
                padding[0, :, self.START_TYPE] = 1  # treat padding nodes as start_type
                g = torch.cat([g, padding], 1)  # 1 * max_n * (n_types + n_vertex)
            if g.shape[2] < self.xs:
                padding = torch.zeros(1, g.shape[1], self.xs-g.shape[2]).to(self.get_device())
                g = torch.cat([g, padding], 2)  # pad zeros to indicate no connections to padding 
                                                # nodes, g: 1 * max_n * xs
            G_new.append(g)
        return torch.cat(G_new, 0)

    def _decode(self, z):
        H0 = self.relu(self.fc3(z))  # batch * hs
        H0_graph = H0.unsqueeze(0).expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
        input_graph_level = self._get_zeros(len(z), self.xs).unsqueeze(1)  # batch * 1 * xs
        type_scores, edge_scores = [], []
        for vi in range(self.max_n):
            output_graph_level, _ = self.grud(input_graph_level, H0_graph)  # batch * 1 * (hs*num_dirs)
            if self.bidir:
                output_graph_level = self.hg_unify(output_graph_level)  # batch * 1 * hs
            H0_graph = output_graph_level.permute(1, 0, 2)  # 1 * batch * hs
            H0_graph = H0_graph.expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
            type_score = self.add_vertex(output_graph_level)  # batch * 1 * nvt
            type_prob = F.softmax(type_score, 2).squeeze(1)  # batch * nvt
            new_type = torch.multinomial(type_prob, 1)  # batch * 1
            type_score = self._one_hot(new_type.reshape(-1).tolist(), self.nvt).unsqueeze(1)  # batch * 1 * nvt
            type_scores.append(type_score)
            H0_edge = output_graph_level.permute(1, 0, 2)  # 1 * batch * hs
            H0_edge = H0_edge.expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
            input_edge_level = self._get_zeros(len(z), 1).unsqueeze(1)  # batch * 1 * 1
            edge_score = []
            for ej in range(self.max_n):
                output_edge_level, _ = self.grud_edge(input_edge_level, H0_edge)  # batch * 1 * (hs*num_dirs)
                if self.bidir:
                    output_edge_level = self.hv_unify(output_edge_level)  # batch * 1 * hs
                H0_edge = output_edge_level.permute(1, 0, 2)
                H0_edge = H0_edge.expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
                edge_score_j = self.sigmoid(self.add_edge(output_edge_level)).detach()  # batch * 1 * 1
                # sample edge j of node i
                edge_score_j = np.random.random_sample((len(z), 1, 1)) < edge_score_j
                edge_score_j = edge_score_j.type(torch.FloatTensor).to(self.get_device())
                edge_score.append(edge_score_j)
                input_edge_level = edge_score_j
            edge_score = torch.cat(edge_score, 2)  # batch * 1 * max_n
            edge_scores.append(edge_score)
            input_graph_level = torch.cat([type_score, edge_score], 2)  # batch * 1 * xs

        type_scores = torch.cat(type_scores, 1)  # batch * max_n * nvt
        edge_scores = torch.cat(edge_scores, 1)  # batch * max_n * max_n

        return type_scores, edge_scores

    def decode(self, z):
        type_scores, edge_scores = self._decode(z)
        # the _decode is already stochastic, so set stochastic=Falsle
        return self.construct_igraph(type_scores, edge_scores, stochastic=False)

    def loss(self, mu, logvar, G_true, beta=0.005):
        # G_true: [batch_size * max_n * xs]
        # use teacher forcing to train (feed groundtruth at each step instead of prediction)
        z = self.reparameterize(mu, logvar)
        H0 = self.relu(self.fc3(z))  # batch * hs
        Input_graph_level = G_true[:, :-1, :].contiguous()
        Input_graph_level = torch.cat([self._get_zeros(len(z), self.xs).unsqueeze(1), Input_graph_level], 1)  # pad initial zeros
        H0_graph = H0.unsqueeze(0).expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
        Output_graph_level, _ = self.grud(Input_graph_level, H0_graph)  # batch * max_n * (hs*num_dirs)
        if self.bidir:
            Output_graph_level = self.hg_unify(Output_graph_level)  # batch * max_n * hs
        type_scores = self.add_vertex(Output_graph_level)  # batch * max_n * nvt

        # merge node dimension with batch dimension as the "new" batch dimension for parallel computing
        H0_edge = Output_graph_level
        H0_edge = H0_edge.reshape(1, -1, self.hs)  # 1 * (batch * max_n) * hs
        H0_edge = H0_edge.expand(self.num_dirs*self.num_layers, -1, -1).contiguous()
        Input_edge_level = G_true[:, :, self.nvt:-1]
        Input_edge_level = torch.cat([self._get_zeros(len(z), self.max_n).unsqueeze(2), Input_edge_level], 2)  # pad initial zeros
        Input_edge_level = Input_edge_level.reshape(-1, self.max_n, 1).contiguous()  # (batch * max_n) * max_n * 1
        Output_edge_level, _ = self.grud_edge(Input_edge_level, H0_edge)  # (batch * max_n) * max_n * (hs * num_dirs)
        if self.bidir:
            Output_edge_level = self.hv_unify(Output_edge_level)  # (batch * max_n) * max_n * hs
        edge_scores = self.sigmoid(self.add_edge(Output_edge_level))  # (batch * max_n) * max_n * 1
        edge_scores = edge_scores.reshape(-1, self.max_n, self.max_n)

        res = 0
        _, true_types = torch.max(G_true[:, :, :self.nvt], 2)
        res += F.cross_entropy(type_scores.transpose(1, 2), true_types, reduction='sum')
        true_edges = G_true[:, :, self.nvt:]
        res += F.binary_cross_entropy(edge_scores, true_edges, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def construct_igraph(self, type_scores, edge_scores, stochastic=True):
        # construct igraphs from node type and edge scores
        # note that when stochastic=True, type_scores should be raw scores before softmax, 
        # and edge_scores should probabilities between [0, 1] (after sigmoid)
        assert(type_scores.shape[:2] == edge_scores.shape[:2])
        if stochastic:
            type_probs = F.softmax(type_scores, 2).cpu().detach().numpy()
        G = []
        for gi in range(len(type_scores)):
            # add vertices
            g = igraph.Graph(directed=True)
            for vj in range(self.max_n):
                if stochastic:
                    new_type = np.random.choice(range(self.nvt), p=type_probs[gi][vj])
                else:
                    new_type = torch.argmax(type_scores[gi][vj], 0).item()
                g.add_vertex(type=new_type)
            # add edges
            output_vertex = None
            for vj in range(self.max_n):
                for ek in range(self.max_n):
                    ek_score = edge_scores[gi][vj][ek].item()
                    if stochastic:
                        if np.random.random_sample() < ek_score:
                            g.add_edge(ek, vj)
                    else:
                        if ek_score > 0.5:
                            g.add_edge(ek, vj)
            # apply a topological order to g (otherwise some validity test doesn't work)
            if g.is_dag():
                topo_order = g.topological_sorting()
                perm = [-1] * len(topo_order)
                for i in range(len(perm)):
                    perm[topo_order[i]] = i
                g = g.permute_vertices(perm) 
            G.append(g)
        return G


'''
    DAG Variational Autoencoder (D-VAE).
'''
class DVAE(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, vid=True):
        super(DVAE, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.vid = vid
        self.device = None

        if self.vid:
            self.vs = hs + max_n  # vertex state size = hidden state + vid
        else:
            self.vs = hs

        # 0. encoding-related
        self.grue_forward = nn.GRUCell(nvt, hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(nvt, hs)  # backward encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar
            
        # 1. decoding-related
        self.grud = nn.GRUCell(nvt, hs)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.ReLU(),
                nn.Linear(hs * 2, nvt)
                )  # which type of new vertex to add f(h0, hg)
        self.add_edge = nn.Sequential(
                nn.Linear(hs * 2, hs * 4), 
                nn.ReLU(), 
                nn.Linear(hs * 4, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew)

        # 2. gate-related
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.gate_backward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False), 
                )

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Sequential(
                    nn.Linear(hs * 2, hs), 
                    )
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.gs * 2, self.gs), 
                    )

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) == list:
            if idx == []:
                return None
            idx = torch.LongTensor([[i] for i in idx])
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.successors(v), self.max_n) for g in G]
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
            gate, mapper = self.gate_forward, self.mapper_forward
        if self.vid:
            H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + 
                            [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) 
                            for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator, reverse=reverse)
        return Hv

    def _update_v(self, G, v, H0=None):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0, reverse=False)
        return
    
    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount()-1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        Hg = self._get_graph_state(G)
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _get_edge_score(self, Hvi, H, H0):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(torch.cat([Hvi, H], 1)))

    def decode(self, z, stochastic=True):
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._update_v(G, 0, H0)
        finished = [False] * len(G)
        for idx in range(1, self.max_n):
            # decide the type of the next added vertex
            if idx == self.max_n - 1:  # force the last node to be end_type
                new_types = [self.END_TYPE] * len(G)
            else:
                Hg = self._get_graph_state(G, decode=True)
                type_scores = self.add_vertex(Hg)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) 
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            for i, g in enumerate(G):
                if not finished[i]:
                    g.add_vertex(type=new_types[i])
            self._update_v(G, idx)

            # decide connections
            edge_scores = []
            for vi in range(idx-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi)
                H = self._get_vertex_state(G, idx)
                ei_score = self._get_edge_score(Hvi, H, H0)
                if stochastic:
                    random_score = torch.rand_like(ei_score)
                    decisions = random_score < ei_score
                else:
                    decisions = ei_score > 0.5
                for i, g in enumerate(G):
                    if finished[i]:
                        continue
                    if new_types[i] == self.END_TYPE: 
                    # if new node is end_type, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                            if v.index != g.vcount()-1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount()-1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi, g.vcount()-1)
                self._update_v(G, idx)

        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory
        return G

    def loss(self, mu, logvar, G_true, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        z = self.reparameterize(mu, logvar)
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
        self._update_v(G, 0, H0)
        res = 0  # log likelihood
        for v_true in range(1, self.max_n):
            # calculate the likelihood of adding true types of nodes
            # use start type to denote padding vertices since start type only appears for vertex 0 
            # and will never be a true type for later vertices, thus it's free to use
            true_types = [g_true.vs[v_true]['type'] if v_true < g_true.vcount() 
                          else self.START_TYPE for g_true in G_true]
            Hg = self._get_graph_state(G, decode=True)
            type_scores = self.add_vertex(Hg)
            # vertex log likelihood
            vll = self.logsoftmax1(type_scores)[np.arange(len(G)), true_types].sum()  
            res = res + vll
            for i, g in enumerate(G):
                if true_types[i] != self.START_TYPE:
                    g.add_vertex(type=true_types[i])
            self._update_v(G, v_true)

            # calculate the likelihood of adding true edges
            true_edges = []
            for i, g_true in enumerate(G_true):
                true_edges.append(g_true.get_adjlist(igraph.IN)[v_true] if v_true < g_true.vcount()
                                  else [])
            edge_scores = []
            for vi in range(v_true-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi)
                H = self._get_vertex_state(G, v_true)
                ei_score = self._get_edge_score(Hvi, H, H0)
                edge_scores.append(ei_score)
                for i, g in enumerate(G):
                    if vi in true_edges[i]:
                        g.add_edge(vi, v_true)
                self._update_v(G, v_true)
            edge_scores = torch.cat(edge_scores[::-1], 1)
            ground_truth = torch.zeros_like(edge_scores)
            idx1 = [i for i, x in enumerate(true_edges) for _ in range(len(x))]
            idx2 = [xx for x in true_edges for xx in x]
            ground_truth[idx1, idx2] = 1.0
            # edges log-likelihood
            ell = - F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum') 
            res = res + ell

        res = -res  # convert likelihood to loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, _, _ = self.loss(mu, logvar, G)
        return loss
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G


'''
    D-VAE with GCN encoder
    The message passing happens at all nodes simultaneously.
'''
class DVAE_GCN(DVAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, levels=1):
        # bidirectional means passing messages ignoring edge directions
        super(DVAE_GCN, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.levels = levels
        self.gconv = nn.ModuleList()
        self.gconv.append(
                nn.Sequential(
                    nn.Linear(nvt, hs), 
                    nn.ReLU(), 
                    )
                )
        for lv in range(1, levels):
            self.gconv.append(
                    nn.Sequential(
                        nn.Linear(hs, hs), 
                        nn.ReLU(), 
                        )
                    )

    def _get_feature(self, g, v, lv=0):
        # get the node feature vector of v
        if lv == 0:  # initial level uses type features
            v_type = g.vs[v]['type']
            x = self._one_hot(v_type, self.nvt)
        else:
            x = g.vs[v]['H_forward']
        return x

    def _get_zero_x(self, n=1):
        # get zero predecessor states X, used for padding
        return torch.zeros(n, self.nvt).to(self.get_device())

    def _get_graph_state(self, G, decode=False, start=0, end_offset=0):
        # get the graph states
        # sum all node states between start and n-end_offset as the graph state
        Hg = []
        max_n_nodes = max(g.vcount() for g in G)
        for g in G:
            hg = torch.cat([g.vs[i]['H_forward'] for i in range(start, g.vcount() - end_offset)],
                           0).unsqueeze(0)  # 1 * n * hs
            if g.vcount() < max_n_nodes:
                hg = torch.cat([hg, 
                    torch.zeros(1, max_n_nodes - g.vcount(), hg.shape[2]).to(self.get_device())],
                    1)  # 1 * max_n * hs
            Hg.append(hg)
        # sum node states as the graph state
        Hg = torch.cat(Hg, 0).sum(1)  # batch * hs
        return Hg  # batch * hs

    def _GCN_propagate_to(self, G, v, lv=0):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return

        if self.bidir:  # ignore edge directions, accept all neighbors' messages
            H_nei = [[self._get_feature(g, v, lv)/(g.degree(v)+1)] + 
                     [self._get_feature(g, x, lv)/math.sqrt((g.degree(x)+1)*(g.degree(v)+1)) 
                     for x in g.neighbors(v)] for g in G]
        else:  # only accept messages from predecessors (generalizing GCN to directed cases)
            H_nei = [[self._get_feature(g, v, lv)/(g.indegree(v)+1)] + 
                     [self._get_feature(g, x, lv)/math.sqrt((g.outdegree(x)+1)*(g.indegree(v)+1)) 
                     for x in g.predecessors(v)] for g in G]
            
        max_n_nei = max([len(x) for x in H_nei])  # maximum number of neighbors
        H_nei = [torch.cat(h_nei + [self._get_zeros(max_n_nei - len(h_nei), h_nei[0].shape[1])], 0).unsqueeze(0) 
                 for h_nei in H_nei]  # pad all to same length
        H_nei = torch.cat(H_nei, 0)  # batch * max_n_nei * nvt
        Hv = self.gconv[lv](H_nei.sum(1))  # batch * hs
        for i, g in enumerate(G):
            g.vs[v]['H_forward'] = Hv[i:i+1]
        return Hv

    def encode(self, G):
        # encode graphs G into latent vectors
        # GCN propagation is now implemented in a non-parallel way for consistency, but
        # can definitely be parallel to speed it up. However, the major computation cost
        # comes from the generation, which is not parallellizable.
        if type(G) != list:
            G = [G]
        prop_order = range(self.max_n)
        for lv in range(self.levels):
            for v_ in prop_order:
                self._GCN_propagate_to(G, v_, lv)
        Hg = self._get_graph_state(G, start=1, end_offset=1)  # does not use the dummy input 
                                                              # and output nodes
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar


'''
    D-VAE for Bayesian networks. 
    The encoding of each node takes gated sum of X instead of H of its predecessors as input.
    The decoding is the same as D-VAE, except for including H0 to predict edge scores.
'''
class DVAE_BN(DVAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False):
        super(DVAE_BN, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.nvt, hs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
                nn.Linear(self.nvt, hs, bias=False), 
                )
        self.gate_forward = nn.Sequential(
                nn.Linear(self.nvt, hs), 
                nn.Sigmoid()
                )
        self.gate_backward = nn.Sequential(
                nn.Linear(self.nvt, hs), 
                nn.Sigmoid()
                )
        self.add_edge = nn.Sequential(
                nn.Linear(hs * 3, hs), 
                nn.ReLU(), 
                nn.Linear(hs, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew, h0)

    def _get_zero_x(self, n=1):
        # get zero predecessor states X, used for padding
        return self._get_zeros(n, self.nvt)

    def _get_graph_state(self, G, decode=False, start=0, end_offset=0):
        # get the graph states
        # sum all node states between start and n-end_offset as the graph state
        Hg = []
        max_n_nodes = max(g.vcount() for g in G)
        for g in G:
            hg = torch.cat([g.vs[i]['H_forward'] for i in range(start, g.vcount() - end_offset)],
                           0).unsqueeze(0)  # 1 * n * hs
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = torch.cat([g.vs[i]['H_backward'] 
                                 for i in range(start, g.vcount() - end_offset)], 0).unsqueeze(0)
                hg = torch.cat([hg, hg_b], 2)
            if g.vcount() < max_n_nodes:
                hg = torch.cat([hg, 
                    torch.zeros(1, max_n_nodes - g.vcount(), hg.shape[2]).to(self.get_device())],
                    1)  # 1 * max_n * hs
            Hg.append(hg)
        # sum node states as the graph state
        Hg = torch.cat(Hg, 0).sum(1)  # batch * hs
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg)
        return Hg  # batch * hs


    def _propagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        # the difference from original D-VAE is using predecessors' X instead of H
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        v_types = [g.vs[v]['type'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[self._one_hot(g.vs[x]['type'], self.nvt) for x in g.successors(v)]
                      for g in G]
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[self._one_hot(g.vs[x]['type'], self.nvt) for x in g.predecessors(v)]
                      for g in G]
            gate, mapper = self.gate_forward, self.mapper_forward
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + 
                          [self._get_zero_x((max_n_pred - len(h_pred)))], 0).unsqueeze(0) 
                          for h_pred in H_pred]
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * hs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        return Hv

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        Hg = self._get_graph_state(G, start=1, end_offset=1)  # does not use the dummy input 
                                                              # and output nodes
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def _get_edge_score(self, Hvi, H, H0):
        # when decoding BN edges, we need to include H0 since the propagation D-separates H0
        # such that Hvi and H do not contain any initial H0 information
        return self.sigmoid(self.add_edge(torch.cat([Hvi, H, H0], 1)))


'''
    A fast D-VAE variant.
    Use D-VAE's encoder + S-VAE's decoder to accelerate decoding.
'''
class DVAE_fast(DVAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False):
        super(DVAE_fast, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.grud = nn.GRU(hs, hs, batch_first=True)  # decoder GRU
        self.add_vertex = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, self.nvt), 
                )
        self.add_edges = nn.Sequential(
                nn.Linear(self.hs, self.hs), 
                nn.ReLU(), 
                nn.Linear(self.hs, self.max_n - 1), 
                )

    def _decode(self, z):
        h0 = self.relu(self.fc3(z))
        h_in = h0.unsqueeze(1).expand(-1, self.max_n - 1, -1)
        h_out, _ = self.grud(h_in)
        type_scores = self.add_vertex(h_out)  # batch * max_n-1 * nvt
        edge_scores = self.sigmoid(self.add_edges(h_out))  # batch * max_n-1 * max_n-1
        return type_scores, edge_scores

    def loss(self, mu, logvar, G_true, beta=0.005):
        # g_true: [batch_size * max_n-1 * xs]
        z = self.reparameterize(mu, logvar)
        type_scores, edge_scores = self._decode(z)
        res = 0
        true_types = torch.LongTensor([[g_true.vs[v_true]['type'] if v_true < g_true.vcount() 
                                      else self.START_TYPE for v_true in range(1, self.max_n)] 
                                      for g_true in G_true]).to(self.get_device())
        res += F.cross_entropy(type_scores.transpose(1, 2), true_types, reduction='sum')
        true_edges = torch.FloatTensor([np.array(g_true.get_adjacency().data).transpose()[1:, :-1]
                                       for g_true in G_true]).to(self.get_device())  # warning! 
                                                    # here all g_true should have the same sizes
        res += F.binary_cross_entropy(edge_scores, true_edges, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld

    def decode(self, z):
        type_scores, edge_scores = self._decode(z)
        return self.construct_igraph(type_scores, edge_scores)

    def construct_igraph(self, type_scores, edge_scores, stochastic=True):
        # copy from S-VAE
        assert(type_scores.shape[:2] == edge_scores.shape[:2])
        if stochastic:
            type_probs = F.softmax(type_scores, 2).cpu().detach().numpy()
        G = []
        for gi in range(len(type_scores)):
            g = igraph.Graph(directed=True)
            g.add_vertex(type=self.START_TYPE)
            for vi in range(1, self.max_n):
                if vi == self.max_n - 1:
                    new_type = self.END_TYPE
                else:
                    if stochastic:
                        new_type = np.random.choice(range(self.nvt), p=type_probs[gi][vi-1])
                    else:
                        new_type = torch.argmax(type_scores[gi][vi-1], 0).item()
                g.add_vertex(type=new_type)
                if new_type == self.END_TYPE:  
                    end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                       if v.index != g.vcount()-1])
                    for v in end_vertices:
                        g.add_edge(v, vi)
                    break
                else:
                    for ej in range(vi):
                        ej_score = edge_scores[gi][vi-1][ej].item()
                        if stochastic:
                            if np.random.random_sample() < ej_score:
                                g.add_edge(ej, vi)
                        else:
                            if ej_score > 0.5:
                                g.add_edge(ej, vi)
            G.append(g)
        return G
