# loss.py: Define the loss functions (here we only need the L1 loss)
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

def my_KLDivLoss(x, y, weights=None):
    """
    from Peng, Han, et al. "Accurate brain age prediction with lightweight deep neural networks." Medical image analysis 68 (2021): 101871.
    Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    if weights is not None:

        loss_func = nn.KLDivLoss(reduction='none')
        y += 1e-16
        loss_unweighted = loss_func(x, y).sum(dim=1)
        loss_weighted = loss_unweighted * weights
        loss = loss_weighted.mean()

    else:
        loss_func = nn.KLDivLoss(reduction='sum')
        y += 1e-16
        n = y.shape[0]
        loss = loss_func(x, y) / n

    return loss

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs.squeeze(), targets.squeeze(), reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


class LabelRegressionNCE(nn.Module):
    def __init__(self, kernel='rbf', temperature=0.1, return_logits=False, sigma=1.5):
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)
        :param temperature:
        :param return_logits:
        """

        # sigma = prior over the label's range
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        if self.kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        else:
            assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

    def forward(self, z, labels, lds_weights=None):
        N = len(z)
        assert N == len(labels), "Unexpected labels length: %i"%len(labels)

        if lds_weights is not None:
            print('weights shape', lds_weights.shape)
            print('z shape', z.shape)
            z *= lds_weights.expand_as(z)

        z = F.normalize(z, p=2, dim=-1) # dim [N, D]

        sim = F.cosine_similarity(z.unsqueeze(0), z.unsqueeze(1), dim=2) / self.temperature

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim = sim - self.INF * torch.eye(N)

        weights = self.kernel(labels.unsqueeze(1), labels.unsqueeze(1))
        weights = weights * (1 - np.eye(N)) # puts 0 on the diagonal

        weights /= weights.sum(axis=1)
        # if 'rbf' kernel and sigma->0, we retrieve the classical NTXenLoss (without labels)
        log_sim = F.log_softmax(sim, dim=1)

        loss = -1. / N * (torch.from_numpy(weights).to(z) * log_sim).sum()

        if self.return_logits:
            return loss, sim

        return loss
