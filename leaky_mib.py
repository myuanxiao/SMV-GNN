# -*- coding: utf-8 -*-
"""
@Time ： 2022/1/27 1:32 下午
@Auth ： wsy
@File ：mib_test.py
@IDE ：PyCharm
@desc ：description
"""
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import torch.optim as optim

class Solver(nn.Module):
    def __init__(self, args, device):
        super(Solver, self).__init__()
        self.encoder_H2d = Encoder(args.z_dim, device)
        self.encoder_H3d = Encoder(args.z_dim,device)
        self.mi_estimator1 = MIEstimator(args.z_dim, args.z_dim, device)
        self.mi_estimator2 = MIEstimator(args.z_dim, args.z_dim, device)
        self.mu = args.mu
        self.eta = args.eta
        self.device = device

        self.to(self.device)

    def _compute_loss(self, H_2d, H_3d):
        p_z2d_given_h2d = self.encoder_H2d(H_2d)  # 这里得到正态分布的参数（mu 和 sigma）
        p_z3d_given_h3d = self.encoder_H3d(H_3d)

        # 重参数技巧   rsample()不是在定义的正太分布上采样，而是先对标准正太分布 N(0,1) 进行采样，然后输出： mean + std × 采样值
        Z_2d = p_z2d_given_h2d.rsample()  # 采样一个z_2d
        Z_3d = p_z3d_given_h3d.rsample()

        # 互信息估计 ==> 对应loss2
        # mi_gradient1 = self.mi_estimator(Z_2d.detach(), Z_3d).mean()  # 这里算出来是负数
        # mi_gradient2 = self.mi_estimator(Z_3d.detach(), Z_2d).mean()
        mi_gradient1 = self.mi_estimator1(H_2d, Z_3d).mean()  # 这里算出来是负数
        mi_gradient2 = self.mi_estimator2(H_3d, Z_2d).mean()
        #mi_gradient1 = self.mi_estimator1(Z_2d, H_3d).mean()  # 这里算出来是负数
        #mi_gradient2 = self.mi_estimator2(Z_3d, H_2d).mean()
        mi_gradient = (mi_gradient1 + mi_gradient2) / 2

        # 离散KL散度需要两个分布中离散元素一致 ==> 对应loss1
        # 分别求 z2d 在 P(z2d|h2d)分布 和 P(z3d|h3d)分布 下真实概率的对数
        kl_23 = p_z2d_given_h2d.log_prob(Z_2d) - p_z3d_given_h3d.log_prob(Z_2d)  # log_prob() 看采样的样本对应分布中的真实概率的对数
        kl_32 = p_z3d_given_h3d.log_prob(Z_3d) - p_z2d_given_h2d.log_prob(Z_3d)
        skl = (kl_23 + kl_32).mean() / 2.

        # 下面这俩是超参
        alpha = self.mu
        beta = self.eta
        loss = - mi_gradient * alpha + beta * skl

        #print(- mi_gradient, skl)
        return loss, Z_2d, Z_3d



class Encoder(nn.Module):
    def __init__(self, z_dim ,device):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, z_dim * 2),  # z的前半段为mu，后半段为sigma
        )
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        params = self.net(x)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive

        return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution


class MIEstimator(nn.Module):
    def __init__(self, size1, size2, device):
        super(MIEstimator, self).__init__()

        # 这里相当于判别器
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 512),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),  # 再试试 (512, 256)
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),  # (256,1)
        )
        self.device = device
        self.to(self.device)

    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples  x2作为anchor
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))  # torch.roll(x1, shifts=1, dim=0)  打乱样本顺序
        # 每个正样本就对应一个负样本
        return -softplus(-pos).mean() - softplus(neg).mean()


