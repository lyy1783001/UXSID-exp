import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "MixLoss"
]

class MixLoss(nn.Module):
    def __init__(
        self, 
        recon_loss: bool = True, 
        orth_loss: bool = False,
        bce_loss: bool = False,
        recon_weight: float = 1.0,
        orth_weight: float = 1.0,
        bce_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.recon_weight = recon_weight
        self.orth_weight = orth_weight
        self.bce_weight = bce_weight

        self.recon_loss = SquareLoss() if recon_loss else None
        self.orth_loss = OrthogonalLoss() if orth_loss else None
        self.bce_loss = torch.functional.F.binary_cross_entropy if bce_loss else None
        
    def forward(self, personalized_vecs, ori_behaviors, recon_behaviors, y_pred=None, y_true=None, enc_weights=None, dec_weights=None, mask=None, reduction='mean'):
        loss_dict = dict()
        
        for weight, loss, name in zip([self.orth_weight], [self.orth_loss], ['orth_loss']):
            if loss is not None and weight > 0:
                loss_dict[name] = weight * loss(personalized_vecs)

        if self.recon_loss is not None and self.recon_weight > 0:
            loss_dict['recon_loss'] = self.recon_weight * self.recon_loss(ori_behaviors, recon_behaviors, mask)
        if self.bce_loss is not None and self.bce_weight > 0:
            loss_dict['bce_loss'] = self.bce_weight * self.bce_loss(y_pred, y_true, reduction=reduction)

        total_loss = sum([v for k, v in loss_dict.items()])
        return total_loss, loss_dict   

    def extra_repr(self):
        loss = []
        for weight, loss, loss_name in zip([self.recon_weight, self.orth_weight, self.bce_weight],
                                            [self.recon_loss, self.orth_loss, self.bce_loss],
                                            ['recon_loss', 'orth_loss', 'bce_loss']):
            if loss is not None and weight > 0:
                loss.append(f"{weight} * {loss_name}")
        
        return f"Loss Constitution: " + " + ".join(loss)
        

class SquareLoss(nn.Module):
    def forward(self, label, pos_score, mask):
        dist = label - pos_score
        dist = dist[~mask]                                                       # mask=true indicates padding
        if label.dim() > 1:
            return torch.mean(torch.mean(torch.square(dist), dim=-1))
        else:
            return torch.mean(torch.square(dist))
        


class OrthogonalLoss(nn.Module):

    def forward(self, x):
        '''x: [B, K, D]'''
        norms = torch.norm(x, dim=-1, keepdim=True)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        x_norm = x / norms
        xy_sim = x_norm @ x_norm.transpose(1, 2)            # (B, K, K)
        eye = torch.eye(x.shape[1], device=x.device)
        penalty = torch.norm(xy_sim - eye, dim=[-2, -1])    # (B)
        loss = torch.mean(penalty)
        return loss



    # def forward(self, weights, mask):
    #     '''
    #     weights: [B, L, K]
    #     mask: [B, L]
    #     '''
    #     n_cluster = weights.shape[-1]
    #     seq_len = (~mask).sum(-1, keepdim=True)                                     # (B)
    #     weights_cp = weights.clone()
    #     weights_cp[mask] = 0
    #     cluster_weights = weights_cp.sum(1) / seq_len                               # (B, K) 
    #     cluster_weights = cluster_weights.sum(0).softmax(0)                         # (K)


    #     uniform_dist = torch.ones_like(cluster_weights) / n_cluster
    #     loss = F.kl_div(cluster_weights.log(), uniform_dist, reduction='sum', log_target=False)
    #     return loss