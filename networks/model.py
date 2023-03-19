import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np
import math
import heapq

from torchdiffeq import odeint

from ode import *

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x

def loss_F_MI(Img, graphnode_ts, graphnode_st, targetss, alpha = 0.5):
    Floss_BCE = nn.BCELoss()
    loss_Img = Floss_BCE(Img,targetss)
    loss_ts = Floss_BCE(graphnode_ts,targetss)
    loss_st = Floss_BCE(graphnode_st,targetss)
    loss_F =  (1-alpha)*loss_Img + alpha*( 0.5*(loss_ts+loss_st) )
    return loss_F



class Discriminator_net(nn.Module):
    def __init__(self, n_hid_first,n_hid_second,out_f_k):
        super(Discriminator_net, self).__init__()
        self.f_k = nn.Bilinear(n_hid_first, n_hid_second, out_f_k)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, graph, node, s_bias1 = None):
        
        D_out = self.f_k(node, graph)
        D_out = D_out[0]
        if s_bias1 is not None:
            D_out += s_bias1
        return D_out


class MergeNet_MLP(nn.Module):   

    def __init__(self, dropout, in_features=2048):
        super(MergeNet_MLP, self).__init__()
        self.dropout = dropout

        self.merge_net = nn.Sequential(nn.Linear(in_features=in_features,
                                                 out_features=2048),
                                       nn.LeakyReLU(0.1),
                                       nn.Dropout(p=dropout),
                                       nn.Linear(in_features=2048,
                                                 out_features=512),
                                       nn.LeakyReLU(0.1),
                                       nn.Dropout(p=dropout),
                                       nn.Linear(in_features=512,
                                       out_features=256),
                                       nn.LeakyReLU(0.1)
                                       )
        
    def forward(self, inputs):
        output = self.merge_net(inputs)
        return output

class GAT_Measures_Net_ode_func(nn.Module):
    def __init__(self, dim_feat, dim_hid, dim_readout, dropout, alpha, nheads, adj):
        """Dense version of GAT."""
        super(GAT_Measures_Net_ode_func, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.dim_hid = dim_hid
        self.dim_readout = dim_readout
        self.adj = adj
        
        self.attentions1 = torch.nn.ModuleList([GraphAttentionLayer(dim_feat, dim_hid, dropout, alpha=alpha) for _ in range(nheads)]) 
        self.attentions2 = torch.nn.ModuleList([GraphAttentionLayer(dim_hid*nheads, dim_readout, dropout, alpha=alpha) for _ in range(nheads)]) 

    def forward(self, t, x):   
        x = torch.cat([att(x, adj=self.adj) for att in self.attentions1], dim=1)
        embedding_nodes = torch.cat([att(x, adj=self.adj) for att in self.attentions2], dim=1)
        
        return embedding_nodes
    
    
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)
        

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    

class Subg_GAT_net_ode(nn.Module):
    def __init__(self, dimfeat_meas, dimhid_meas, dimreadout_meas, nheads_meas,
                 dropout, alpha, device, 
                 dim_hid_edge, nheads_edge, full_seg_adj):
        super(Subg_GAT_net_ode, self).__init__()
        self.full_seg_adj = full_seg_adj
        N = self.full_seg_adj.size(0)
        self.odeint = odeint
        tol_scale = 1.0
        self.atol = tol_scale * 1e-3
        self.rtol = tol_scale * 1e-3
        
        self.odefunc = GAT_Measures_Net_ode_func(512, dimhid_meas, dimreadout_meas//nheads_meas, dropout, alpha, nheads_meas, self.full_seg_adj)
        
        self.method = 'euler'
        self.step_size = 1.0
        self.t = torch.tensor([0, 1], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.ImageFeature = CNN_ode(downsampling_method='res')
        self.dropout = dropout
        
    def forward(self, x_meas_src):
        seg_from_image_src = self.ImageFeature(x_meas_src).view(x_meas_src.size(0), -1)
        seg_src = seg_from_image_src
    
        t = self.t.type_as(seg_src).cuda()
        integrator = self.odeint
        func = self.odefunc
        state = seg_src
        state_dt = integrator(
            func, state, t,
            atol=self.atol,
            rtol=self.rtol)
        z = state_dt[1]
        embedding_graph_src = z
        
        return embedding_graph_src

class Match_GAT_MI(nn.Module):
    def __init__(self, dimfeat_meas, dimhid_meas, dimreadout_meas, nheads_meas,
                 dropout, alpha, device, 
                 dim_hid_edge, nheads_edge, full_seg_adj):
        super(Match_GAT_MI, self).__init__()
        
        self.dropout = dropout
        
        self.Subg_GAT = Subg_GAT_net_ode(dimfeat_meas, dimhid_meas, dimreadout_meas, nheads_meas, dropout, alpha, device, dim_hid_edge, nheads_edge, full_seg_adj)     
        self.Image_CNN = CNN_ode(downsampling_method='res')
        self.MLPnn = nn.Sequential(nn.Linear(512, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 1024*2))  

        self.Discriminator_MI = Discriminator_net(512,512,1)
        self.Merge = MergeNet_MLP(dropout, 1024*2)
        self.fc = nn.Linear(256,1)
        self.fout = nn.Sigmoid()       
        
    def forward(self, x_meas_src, x_meas_tar, x_subg_tf_src, x_subg_tf_tar):
        
        x_subg_meas_src = x_subg_tf_src
        x_subg_meas_tar = x_subg_tf_tar
        
        GAT_subgraph_src = self.Subg_GAT(x_subg_meas_src)
        GAT_subgraph_tar = self.Subg_GAT(x_subg_meas_tar)
        
        embedding_Image_src = self.Image_CNN(x_meas_src).view(x_meas_src.size(0), -1)
        embedding_Image_tar = self.Image_CNN(x_meas_tar).view(x_meas_tar.size(0), -1)
        
        embedding_subgraph_src = torch.max(GAT_subgraph_src, dim=0)[0].view(x_meas_src.size(0), -1)
        embedding_subgraph_tar = torch.max(GAT_subgraph_tar, dim=0)[0].view(x_meas_src.size(0), -1)

        N_embedding = torch.pow(embedding_Image_src-embedding_Image_tar, 2)
        N_embedding = self.MLPnn(N_embedding)
        N_embedding = torch.max(N_embedding, dim=0)[0]
        output_Image = self.Merge(N_embedding)
        output_Image = self.fc(output_Image)
        output_Image = self.fout(output_Image)
        
        D_out_ts = self.Discriminator_MI(embedding_subgraph_tar, embedding_Image_src)
        D_out_st = self.Discriminator_MI(embedding_subgraph_src, embedding_Image_tar)

        out_ts = self.fout(D_out_ts)
        out_st = self.fout(D_out_st)

        return output_Image, out_ts, out_st

