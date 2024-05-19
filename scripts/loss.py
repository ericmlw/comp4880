import torch
import torch.nn as nn
import numpy as np

class Loss(nn.Module):
    def __init__(self, mask_qi, mask_it, laplacian_loss_weight):
        super(Loss, self).__init__()
        self.mask_qi = mask_qi
        self.mask_it = mask_it
        
        self.num_qi = float(mask_qi.sum())
        self.num_it = float(mask_it.sum())
        
        self.logsm = nn.LogSoftmax(dim=0)
        self.sm = nn.Softmax(dim=0)
        self.laplacian_loss_weight = laplacian_loss_weight
        
    def cross_entropy(self, score, mask):
        l = torch.sum(-mask * self.logsm(score))
        return l / mask.sum()
    
    def binary_cross_entropy(self, score, mask):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(score, mask.float())
        return loss
    
    def rmse(self, score, mask):
        pred = torch.sigmoid(score)
        mse = torch.sum((pred * mask - mask) ** 2) / mask.sum()
        return torch.sqrt(mse)
    
    def loss(self, score_qi, score_it):
        return self.binary_cross_entropy(score_qi, self.mask_qi) + self.binary_cross_entropy(score_it, self.mask_it)
    
    def laplacian_loss(self, score_qi, score_it, laplacian_q, laplacian_i, laplacian_t):
        pred_qi = torch.sigmoid(score_qi)
        pred_it = torch.sigmoid(score_it)
        
        dirichlet_q = torch.mm(torch.mm(torch.transpose(pred_qi, 0, 1), laplacian_q.float()), pred_qi.float())
        dirichlet_i = torch.mm(torch.mm(torch.transpose(pred_it, 0, 1), laplacian_i.float()), pred_it.float())
        
        dirichlet_norm_q = torch.trace(dirichlet_q) / (laplacian_q.shape[0] * laplacian_q.shape[1])
        dirichlet_norm_i = torch.trace(dirichlet_i) / (laplacian_i.shape[0] * laplacian_i.shape[1])
        
        return self.laplacian_loss_weight * (dirichlet_norm_q + dirichlet_norm_i) + (1 - self.laplacian_loss_weight) * self.loss(score_qi, score_it)
