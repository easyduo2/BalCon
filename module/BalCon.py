# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
L2norm = nn.functional.normalize

import numpy as np
import copy


from .preprocess import *
from .loss_fun import *

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss
    
class BalCon(nn.Module):
    def __init__(self, feats_dim, sub_num, hidden_dim, embed_dim, num_clusters, tau, dropout, nnodes, dominant_index, nlayer, device, alpha=0.7, beta=0.5):
        super(BalCon, self).__init__()
        self.feats_dim = feats_dim
        self.embed_dim = embed_dim
        self.sub_num = sub_num
        self.tau = tau
        self.device = device
        self.nlayer = nlayer
        self.num_clusters = num_clusters
        self.dominant_index = dominant_index
        self.alpha = alpha
        self.beta = beta
        self.online_encoder = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(feats_dim, hidden_dim),
                                nn.ELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, embed_dim),
                                )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.decoder = nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Linear(embed_dim, hidden_dim),
                                nn.ELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, feats_dim),
                                )
        
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

        self.cl = ContrastiveLoss(self.tau)

    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)

    def forward(self, feat, f_list, re_computing=False, warm_up=True):
        self._update_target_branch(0.99)
        z_list = [self.online_encoder(f_list[i]) for i in range(self.sub_num)]
        dominant_index, err_loss = self.dominant_view_mining(z_list, feat, warm_up)
        if re_computing:
            self.dominant_index = dominant_index
        z_t = [self.target_encoder(f_list[i]) for i in range(self.sub_num)]

        if warm_up:
            mp = torch.eye(z_t[0].shape[0]).cuda()
            mp = [mp for _ in range(self.sub_num)]
        else:
            mp = [self.kernel_affinity(z_t[i]) for i in range(self.sub_num)]

        """contrastive loss with projection"""
        # proj_list = [self.proj[i](z_list[i]) for i in range(self.sub_num)]
        
        l_intra = torch.sum(torch.stack([self.cl(z_list[i], z_t[i], mp[i]) for i in range(self.sub_num)])) / self.sub_num
        l_inter = 0.
        for i in range(self.sub_num):
            if i != self.dominant_index:
                l_inter += self.cl(z_list[i], z_t[self.dominant_index], mp[self.dominant_index])
        l_inter = l_inter / (self.sub_num - 1)
        loss_contrastive = self.alpha * l_inter + (1 - self.alpha) * l_intra

        """reconstruction loss"""
        r_list = [self.decoder(z_list[i]) for i in range(self.sub_num)]
        ae_loss = 0.
        for i in range(self.sub_num):
            ae_loss += sce_loss(r_list[i], f_list[i])
        ae_loss = ae_loss / self.sub_num

        """clustering loss"""
        clu_loss = KL_clustering(self.device, z_list, self.num_clusters, self.dominant_index)

        loss = loss_contrastive + ae_loss + self.beta*err_loss + clu_loss * 100

        result = {'total_loss': loss.item(), 'ae_loss': ae_loss.item(), 'err_loss': err_loss.item(), 'clu_loss': clu_loss.item(), 'contrastive_loss': loss_contrastive.item()}
        return loss, result

    def forward_discriminator(self, feat, f_list):
        z_list = [self.target_encoder(f_list[i]) for i in range(self.sub_num)]
        z_true = self.target_encoder(feat)

        batch_size = [i.shape[0] for i in z_list]
        true_labels = torch.ones(z_true.shape[0]).to(self.device)
        false_labels = [torch.zeros(i).to(self.device) for i in batch_size]

        # calculate the loss of true samples
        true_data = self.discriminator(z_true).reshape(-1)
        loss_true = F.binary_cross_entropy(true_data, true_labels)

        # calculate the loss of false samples
        false_data = [self.discriminator(z_list[i]).reshape(-1) for i in range(self.sub_num)]
        loss_false = [F.binary_cross_entropy(false_data[i], false_labels[i]) for i in range(self.sub_num)]

        # total loss
        loss = (loss_true + torch.sum(torch.stack(loss_false)) / self.sub_num) / 2
        return loss

    @torch.no_grad()
    def kernel_affinity(self, z, temperature=0.1, step: int = 5):
        z = L2norm(z)
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)
        G = torch.exp(-G / temperature)
        G = G / G.sum(dim=1, keepdim=True)

        G = torch.matrix_power(G, step)
        alpha = 0.5
        G = torch.eye(G.shape[0]).cuda() * alpha + G * (1 - alpha)
        return G
  
    def dominant_view_mining(self, z_list, feat, warm_up):
        # calculate the similarity matrix of original features
        feat = F.normalize(feat, dim=1, p=2)
        feat_sim = torch.mm(feat, feat.t())
        mining_lambda = 0.5 if warm_up else 0.1
        # get the output of discriminator for quality evaluation
        batch_size = [i.shape[0] for i in z_list]
        true_labels = [torch.ones(i).to(self.device) for i in batch_size]  # change to ones
        adv_outputs = [self.discriminator(z_list[i]).reshape(-1) for i in range(len(z_list))]
        
        # calculate the quality score of each view
        quality_scores = []
        for i in range(len(z_list)):
            # discriminator score: the smaller the better
            disc_score = F.binary_cross_entropy(adv_outputs[i], true_labels[i])
            
            # calculate the MSE of the similarity matrix of original features
            z_norm_i = F.normalize(z_list[i], dim=1, p=2)
            z_sim = torch.mm(z_norm_i, z_norm_i.t())
            mse_score = F.mse_loss(z_sim, feat_sim)
            
            # comprehensive score: the smaller the better
            quality_scores.append(disc_score + mining_lambda * mse_score)
        
        # select the view with the lowest score as the dominant view
        dominant_index = torch.argmin(torch.tensor(quality_scores))
        err_loss = sum(quality_scores) / len(quality_scores)
        
        return dominant_index, err_loss

    @torch.no_grad()
    def get_embeds(self, f_list):
        z_list = [self.target_encoder(f_list[i]) for i in range(self.sub_num)]
        z = torch.cat(z_list, dim=1)
        z = F.normalize(z, dim=1, p=2)
        return z.detach()
