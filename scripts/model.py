#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:02:15 2019

@author: YuxuanLong
"""

import torch
import torch.nn as nn
import torch.sparse as sp
import numpy as np
import utils
from loss import Loss

def sparse_drop(feature, drop_out):
    tem = torch.rand((feature._nnz()))
    feature._values()[tem < drop_out] = 0
    return feature

class GCMC(nn.Module):
    def __init__(self, feature_q,feature_i, feature_t, feature_dim, hidden_dim, M_qi, M_it, out_dim, drop_out = 0.0):
        super(GCMC, self).__init__()
        ###To Do:
        #### regularization on Q
        
        self.drop_out = drop_out
        

        self.feature_q = feature_q
        self.feature_i = feature_i
        self.feature_t = feature_t

        
        self.num_q = feature_q.shape[0]
        self.num_i = feature_i.shape[0]
        self.num_t = feature_t.shape[0]

        

        self.W = nn.Parameter(torch.randn(feature_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W, mode = 'fan_out', nonlinearity = 'relu')
        
        self.M_qi = M_qi
        self.M_it = M_it
        
        self.reLU = nn.ReLU()
        

        self.linear_cat_q = nn.Sequential(*[nn.Linear(hidden_dim * 2, out_dim, bias = True), 
                                            nn.BatchNorm1d(out_dim), nn.ReLU()])
        self.linear_cat_i = nn.Sequential(*[nn.Linear(hidden_dim * 2, out_dim, bias = True), 
                                            nn.BatchNorm1d(out_dim), nn.ReLU()])
        self.linear_cat_t = nn.Sequential(*[nn.Linear(hidden_dim * 2, out_dim, bias = True), 
                                            nn.BatchNorm1d(out_dim), nn.ReLU()])


        self.Q = nn.Parameter(torch.randn(out_dim, out_dim))
        nn.init.orthogonal_(self.Q)
        
        
    def forward(self):
        
        feature_q_drop = sparse_drop(self.feature_q, self.drop_out) / (1.0 - self.drop_out)
        feature_i_drop = sparse_drop(self.feature_i, self.drop_out) / (1.0 - self.drop_out)
        feature_t_drop = sparse_drop(self.feature_t, self.drop_out) / (1.0 - self.drop_out)

        #M_qi is row q column i, hidden u is row instance column feature
        hidden_q = sp.mm(feature_i_drop, self.W)
        hidden_q = self.reLU(sp.mm(self.M_qi, hidden_q))
        
        ### need to further process M, normalization
        hidden_i = sp.mm(feature_q_drop, self.W)
        hidden_i_temp = sp.mm(feature_t_drop, self.W)
        hidden_i = self.reLU(sp.mm(self.M_qi.T, hidden_i)+sp.mm(self.M_it, hidden_i_temp)) 

        hidden_t = sp.mm(feature_i_drop, self.W)
        hidden_t = self.reLU(sp.mm(self.M_it.T, hidden_t))        


        cat_q = torch.cat((hidden_q, self.feature_q), dim=1)
        cat_i = torch.cat((hidden_i, self.feature_i), dim=1)
        cat_t = torch.cat((hidden_t, self.feature_t), dim=1)


        embed_q = self.linear_cat_q(cat_q)
        embed_i = self.linear_cat_i(cat_i)
        embed_t = self.linear_cat_t(cat_t)

        

        #decoder
        score_qi = torch.mm(torch.mm(embed_q, self.Q), torch.t(embed_i))
        score_it = torch.mm(torch.mm(embed_i, self.Q), torch.t(embed_t))

        
        return score_qi,score_it
    

        


