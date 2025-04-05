import torch
import torch.nn as nn
from util import calculate_dynamic_NN_nb


class DynLocRep_loss(nn.Module):
    """Supervised contrastive loss: https://arxiv.org/pdf/2004.11362.pdf.
    Based on: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, method: str, temperature: float = 0.07, contrast_mode: str = 'all',
                 base_temperature: float = 0.07, kernel: callable = None, delta_reduction: str = 'sum',
                 epochs: int = 200, NN_nb_step_size: int = 0, end_NN_nb: int = 4, NN_nb_selection: str = 'similarity'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.method = method
        self.kernel = kernel
        self.delta_reduction = delta_reduction
        self.epochs = epochs

        # new hyperparams for modifications
        self.NN_nb_step_size = NN_nb_step_size
        self.end_NN_nb = end_NN_nb
        self.NN_nb_selection = NN_nb_selection

        if kernel is not None and method == 'supcon':
            raise ValueError('Kernel must be none if method=supcon')

        if kernel is None and method != 'supcon':
            raise ValueError('Kernel must not be none if method != supcon')

        if delta_reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction {delta_reduction}")

    def __repr__(self):
        return f'{self.__class__.__name__} ' \
               f'(t={self.temperature}, ' \
               f'method={self.method}, ' \
               f'kernel={self.kernel is not None}, ' \
               f'delta_reduction={self.delta_reduction})'

    def forward(self, features, labels=None, epoch=None):
        """Compute loss for model. If `labels` is None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, n_features].
                input has to be rearranged to [bsz, n_views, n_features] and labels [bsz],
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) != 3:
            raise ValueError('`features` needs to be [bsz, n_views, n_feats],'
                             '3 dimensions are required')

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is None:
            mask = torch.eye(batch_size, device=device)

        else:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            if self.kernel is None:
                mask = torch.eq(labels, labels.T)
            else:
                mask = self.kernel(labels)

        view_count = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            features = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            features = features
            anchor_count = view_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Tile mask
        mask = mask.repeat(anchor_count, view_count)

        # Inverse of torch-eye to remove self-contrast (diagonal)
        inv_diagonal = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * n_views, device=device).view(-1, 1),
            0
        )

        # compute similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # dynamic localised repulsion
        if self.NN_nb_step_size > 0 and batch_size >= 32:
            dynamic_NN_nb = calculate_dynamic_NN_nb(current_epoch=epoch,
                                                    max_epochs=self.epochs,
                                                    start_NN_nb=batch_size,
                                                    end_NN_nb=self.end_NN_nb,
                                                    step_size=self.NN_nb_step_size)
            if self.NN_nb_selection == 'euclidean':
                _, indices = torch.topk(torch.div(1, torch.cdist(features, features)),
                                        n_views * dynamic_NN_nb, dim=1)
            elif self.NN_nb_selection == 'similarity':
                _, indices = torch.topk(logits, n_views * dynamic_NN_nb, dim=1)
            elif self.NN_nb_selection == 'manhattan':
                _, indices = torch.topk(torch.div(1, torch.cdist(features, features, p=1)),
                                        n_views * dynamic_NN_nb, dim=1)
            elif self.NN_nb_selection == 'chebyshev':
                _, indices = torch.topk(torch.div(1, torch.cdist(features, features, p=float('inf'))),
                                        n_views * dynamic_NN_nb, dim=1)
            else:
                raise ValueError(f"Invalid NN_nb_selection {self.NN_nb_selection}")

            neighbor_mask = torch.zeros_like(logits).scatter_(1, indices, 1)

        else:
            # If neighborhood_size is not set, use a mask that includes all entries
            neighbor_mask = torch.ones_like(logits)

        alignment_logits = logits
        uniformity_logits = logits

        alignment_mask = mask
        uniformity_mask = mask

        if self.method == 'expw':
            # exp weight e^(s_j(1-w_j))

            distance_weighting = (1 - uniformity_mask)

            uniformity = torch.exp(uniformity_logits * distance_weighting) * neighbor_mask * inv_diagonal

        # base case is:
        # - supcon if kernel = none
        # - y-aware is kernel != none
        else:
            uniformity = torch.exp(logits) * inv_diagonal

        if self.method == 'threshold':
            repeated = mask.unsqueeze(-1).repeat(1, 1, mask.shape[0])  # repeat kernel mask

            delta = (mask[:, None].T - repeated.T).transpose(1, 2)  # compute the difference w_k - w_j for every k,j
            delta = (delta > 0.).float()

            # for each z_i, repel only samples j s.t. K(z_i, z_j) < K(z_i, z_k)
            uniformity = uniformity.unsqueeze(-1).repeat(1, 1, mask.shape[0])

            if self.delta_reduction == 'mean':
                uniformity = (uniformity * delta).mean(-1)
            else:
                uniformity = (uniformity * delta).sum(-1)

        uniformity = torch.log(uniformity.sum(1, keepdim=True))

        # positive mask contains the anchor-positive pairs
        # excluding <self,self> on the diagonal
        positive_mask = alignment_mask * inv_diagonal

        log_prob = alignment_logits - uniformity  # log(alignment/uniformity) = log(alignment) - log(uniformity)

        log_prob = (positive_mask * log_prob).sum(1)
        loss = -torch.sum(log_prob) / torch.sum(positive_mask)

        loss = (self.temperature / self.base_temperature) * loss
        contrast_loss = loss.mean()

        total_loss = contrast_loss

        return total_loss


class KernelizedSupCon(nn.Module):
    """from https://github.com/EIDOSLAB/contrastive-brain-age-prediction"""

    def __init__(self, method: str, temperature: float = 0.07, contrast_mode: str = 'all',
                 base_temperature: float = 0.07, kernel: callable = None, delta_reduction: str = 'sum'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.method = method
        self.kernel = kernel
        self.delta_reduction = delta_reduction

        if kernel is not None and method == 'supcon':
            raise ValueError('Kernel must be none if method=supcon')

        if kernel is None and method != 'supcon':
            raise ValueError('Kernel must not be none if method != supcon')

        if delta_reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction {delta_reduction}")

    def __repr__(self):
        return f'{self.__class__.__name__} ' \
               f'(t={self.temperature}, ' \
               f'method={self.method}, ' \
               f'kernel={self.kernel is not None}, ' \
               f'delta_reduction={self.delta_reduction})'

    def forward(self, features, labels=None):
        """Compute loss for model. If `labels` is None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, n_features].
                input has to be rearranged to [bsz, n_views, n_features] and labels [bsz],
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) != 3:
            raise ValueError('`features` needs to be [bsz, n_views, n_feats],'
                             '3 dimensions are required')

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is None:
            mask = torch.eye(batch_size, device=device)

        else:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            if self.kernel is None:
                mask = torch.eq(labels, labels.T)
            else:
                mask = self.kernel(labels)

        view_count = features.shape[1]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            features = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            features = features
            anchor_count = view_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Tile mask
        mask = mask.repeat(anchor_count, view_count)

        # Inverse of torch-eye to remove self-contrast (diagonal)
        inv_diagonal = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * n_views, device=device).view(-1, 1),
            0
        )

        # compute similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        alignment = logits

        # base case is:
        # - supcon if kernel = none
        # - y-aware is kernel != none
        uniformity = torch.exp(logits) * inv_diagonal

        if self.method == 'threshold':
            repeated = mask.unsqueeze(-1).repeat(1, 1, mask.shape[0])  # repeat kernel mask

            delta = (mask[:, None].T - repeated.T).transpose(1, 2)  # compute the difference w_k - w_j for every k,j
            delta = (delta > 0.).float()

            # for each z_i, repel only samples j s.t. K(z_i, z_j) < K(z_i, z_k)
            uniformity = uniformity.unsqueeze(-1).repeat(1, 1, mask.shape[0])

            if self.delta_reduction == 'mean':
                uniformity = (uniformity * delta).mean(-1)
            else:
                uniformity = (uniformity * delta).sum(-1)

        elif self.method == 'expw':
            # exp weight e^(s_j(1-w_j))
            uniformity = torch.exp(logits * (1 - mask)) * inv_diagonal

        uniformity = torch.log(uniformity.sum(1, keepdim=True))

        # positive mask contains the anchor-positive pairs
        # excluding <self,self> on the diagonal
        positive_mask = mask * inv_diagonal

        log_prob = alignment - uniformity  # log(alignment/uniformity) = log(alignment) - log(uniformity)
        log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(
            1)  # compute mean of log-likelihood over positive

        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob
        return loss.mean()