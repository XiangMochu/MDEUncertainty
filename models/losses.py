import random
import torch 
from torch import nn
from torch.nn import functional as F
from einops import rearrange
# import torchsort


class RegLoss:
    ''' pred_depth, gt '''
    def __init__(self, spv_type):
        self.spv_type = spv_type

        loss_selection = {
            'regression_l1': regression_l1_loss,
            'regression_rms': regression_rms_loss,
            'regression_silog_loss': regression_silog_loss,
            'none': none_loss
            }
        self.loss_fn = loss_selection[spv_type]
    
    def __call__(self, **kwargs):
        if 'weight' in kwargs.keys():
            weight = kwargs['weight']
            kwargs.__delitem__('weight')
            loss = (weight * self.loss_fn(**kwargs))
        else:
            loss = self.loss_fn(**kwargs)
        loss = loss.mean() if isinstance(loss, torch.Tensor) else loss
        loss = loss.sqrt() if self.spv_type == 'regression_rms' else loss
        return loss


class ProbLoss:
    ''' pred_prob, scales, gt '''
    def __init__(self, spv_type):
        self.spv_type = spv_type
        loss_selection = {
            'soft_label': soft_label_loss,
            'soft_label_nll': soft_label_nll_loss,
            'stochastic_soft_label': stochastic_soft_label_loss,
            'hard_label': hard_label_loss,
            'hard_label_entropy': hard_label_entropy_loss,
            'confidence_weighted_rmse': confidence_weighted_rmse_loss,
            'confidence_weighted_l1': confidence_weighted_l1_loss,
            'uncertainty_softened_label': uncertainty_softened_label_loss,
            'error_softened_label': error_softened_label_loss,
            'none': none_loss
            }
        self.loss_fn = loss_selection[spv_type]
    
    def __call__(self, **kwargs):
        if 'weight' in kwargs.keys():
            weight = kwargs['weight']
            kwargs.__delitem__('weight')
            loss = (weight * self.loss_fn(**kwargs)).mean()
        else:
            loss = self.loss_fn(**kwargs).mean()
        return loss


class UncertLoss:
    '''  uncertainty, pred_depth, gt '''
    def __init__(self, spv_type):
        self.spv_type = spv_type
        loss_selection = {
            'entropy': entropy_loss,
            'error_uncertainty_ranking': error_uncertainty_ranking_loss,
            'error_uncertainty_ranking_noclamp': error_uncertainty_ranking_loss_noclamp,
            'error_uncertainty_spearman': error_uncertainty_spearman_loss,
            'error_uncertainty_correlation': error_uncertainty_correlation_loss,
            'error_uncertainty_l1': error_uncertainty_l1_loss,
            'alex_kendall_reweight': alex_kendall_reweight_loss,
            'none': none_loss
            }
        self.loss_fn = loss_selection[spv_type]
    
    def __call__(self, **kwargs):
        if 'weight' in kwargs.keys():
            weight = kwargs['weight']
            kwargs.__delitem__('weight')
            loss = (weight * self.loss_fn(**kwargs)).mean()
        else:
            loss = self.loss_fn(**kwargs).mean()
        return loss

    
def stochastic_soft_label_loss(pred_prob, scales, gt):
    gt_label = depth_to_stochastic_soft_label(gt, scales)
    clas_loss = torch.abs(pred_prob - gt_label)
    return clas_loss


def soft_label_loss(pred_prob, scales, gt):
    gt_label = depth_to_soft_label(gt, scales)
    clas_loss = torch.abs(pred_prob - gt_label)
    return clas_loss


def soft_label_nll_loss(pred_prob, scales, gt):
    gt_label = depth_to_soft_label(gt, scales)
    clas_loss = - gt_label * torch.log(pred_prob + 1e-8) - (1 - gt_label) * torch.log(1 - pred_prob + 1e-8)
    return clas_loss


def hard_label_entropy_loss(pred_prob, scales, gt):
    gt_label = depth_to_hard_label_onehot(gt, scales)
    clas_loss = F.cross_entropy(pred_prob, gt_label, reduce=False)
    return clas_loss


def hard_label_loss(pred_prob, scales, gt):
    gt_label = depth_to_hard_label_onehot(gt, scales)
    clas_loss = torch.abs(pred_prob - gt_label)
    return clas_loss


def confidence_weighted_rmse_loss(pred_prob, scales, gt):
    scales = rearrange(scales, 'd -> 1 d 1 1')
    errors = torch.pow((scales - gt), 2)
    weighted = pred_prob * errors
    return weighted


def confidence_weighted_l1_loss(pred_prob, scales, gt):
    scales = rearrange(scales, 'd -> 1 d 1 1')
    errors = torch.abs((scales - gt))
    weighted = pred_prob * errors
    return weighted


def regression_l1_loss(pred_depth, gt):
    errors = torch.abs(pred_depth - gt)
    return errors

def regression_rms_loss(pred_depth, gt):
    errors = torch.pow(pred_depth - gt, 2)
    return errors

def entropy_loss(pred_prob, scales, gt):
    return (-torch.log(pred_prob + 1e-8) * pred_prob) / 1e3


# from AdaBins
def regression_silog_loss(pred_depth, gt):
    g = torch.log(pred_depth) - torch.log(gt)
    Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
    errors = 10 * torch.sqrt(Dg)
    return errors


def none_loss(*args, **kwargs):
    return 0


def uncertainty_softened_label_loss(pred_prob, scales, gt, confidence):
    # scale_temp = 10 + confidence * 20 # [0,1] -> [10, 30]
    scale_temp = 15 + confidence * 10 # [0,1] -> [15, 25]
    # scale_temp = 10 + confidence * 10 # [0,1] -> [10, 20]
    gt_label = depth_to_stochastic_soft_label(gt, scales, scale_temp)
    clas_loss = torch.abs(pred_prob - gt_label)
    return clas_loss


def error_softened_label_loss(pred_prob, scales, gt, pred_depth):
    error = torch.abs(pred_depth - gt).detach()
    scale_temp = error / error.max() # 0-1 normalizing
    gt_label = depth_to_stochastic_soft_label(gt, scales, scale_temp)
    clas_loss = torch.abs(pred_prob - gt_label.detach())
    return clas_loss


def error_uncertainty_ranking_loss(uncertainty, pred_depth, gt):
    error = (pred_depth - gt).abs()
    if len(pred_depth.shape) == 4:
        error = rearrange(error, 'b c h w -> b (c h w)')
        uncertainty = rearrange(uncertainty, 'b c h w -> b (c h w)')
    randperm_idx = torch.randperm(error.shape[-1])
    error_ = error.clone()[..., randperm_idx]
    uncertainty_ = uncertainty.clone()[..., randperm_idx]

    ranking_loss = ((error-error_).detach() - (uncertainty-uncertainty_)).clamp(0).mean()
    return ranking_loss


def error_uncertainty_l1_loss(uncertainty, pred_depth, gt):
    error = (pred_depth - gt).abs()
    if len(pred_depth.shape) == 4:
        error = rearrange(error, 'b c h w -> b (c h w)')
        uncertainty = rearrange(uncertainty, 'b c h w -> b (c h w)')
    l1_loss = (error.detach() - uncertainty).abs().mean()
    return l1_loss


def error_uncertainty_ranking_loss_noclamp(uncertainty, pred_depth, gt):
    error = (pred_depth - gt).abs()
    if len(pred_depth.shape) == 4:
        error = rearrange(error, 'b c h w -> b (c h w)')
        uncertainty = rearrange(uncertainty, 'b c h w -> b (c h w)')
    randperm_idx = torch.randperm(error.shape[-1])
    error_ = error.clone()[..., randperm_idx]
    uncertainty_ = uncertainty.clone()[..., randperm_idx]

    ranking_loss = ((error-error_).detach() - (uncertainty-uncertainty_)).mean()
    return ranking_loss


def error_uncertainty_spearman_loss(uncertainty, pred_depth, gt):
    error = (pred_depth - gt).abs()
    uncertainty = rearrange(uncertainty, 'b c h w -> b (c h w)')
    error = rearrange(error, 'b c h w -> b (c h w)')
    uncertainty = torchsort.soft_rank(uncertainty)
    error = torchsort.soft_rank(error)

    uncertainty = uncertainty - uncertainty.mean()
    uncertainty = uncertainty / uncertainty.norm()

    error = error - error.mean()
    error = error / error.norm()

    spearman_loss = -(uncertainty * error.detach()).mean()
    return spearman_loss


def error_uncertainty_correlation_loss(uncertainty, pred_depth, gt):
    error = (pred_depth - gt).abs()
    uncertainty = rearrange(uncertainty, 'b c h w -> b (c h w)')
    error = rearrange(error, 'b c h w -> b (c h w)')

    uncertainty = uncertainty / uncertainty.std(-1, keepdim=True)
    error = error / error.std(-1, keepdim=True)

    correlation_loss = torch.exp(-(uncertainty * error.detach()).mean())
    return correlation_loss
        

def depth_to_hard_label(gt, scale):
    '''
    convert GT depth map to hard label map
    gt : b 1 h w
    scale : d
    label : b h w
    '''
    scale = rearrange(scale, 'd -> 1 d 1 1')
    label = torch.argmin(torch.abs(gt - scale), dim=1)
    return label


def depth_to_hard_label_onehot(gt, scale):
    '''
    convert GT depth map to hard label map
    gt : b 1 h w
    scale : d
    label : b d h w
    '''
    scale = rearrange(scale, 'd -> 1 d 1 1')
    cost = torch.abs(gt - scale)
    label = cost == torch.min(cost, dim=1, keepdim=True)[0]
    return label.to(torch.float)


def _depth_to_soft_label(gt, scale):
    '''
    convert GT depth map to soft label map
    gt : b 1 h w
    scale : d
    label : b d h w
    '''
    WINDOW_THRESH = 0.1
    SCALE_FACTOR = 20

    scale = rearrange(scale, 'd -> 1 d 1 1')
    l1_distance = torch.abs(gt - scale)
    # l2_distance = torch.pow((gt-scale), 2)
    # exp_distance = torch.exp(torch.abs(gt - scale))
    window = l1_distance < WINDOW_THRESH

    cost = torch.exp(-l1_distance*SCALE_FACTOR) * window
    soft_label = cost / cost.sum(1, keepdim=True)

    return soft_label


def depth_to_soft_label(gt, scale):
    '''
    convert GT depth map to soft label map
    gt : b 1 h w | n
    scale : d
    label : b d h w | d n
    '''
    WINDOW_RADIUS = 2
    SCALE_FACTOR = 20
    if len(gt.shape) == 1:
        gt = rearrange(gt, 'n -> 1 n')
        scale = rearrange(scale, 'd -> d 1')
    elif not len(scale.shape) == 4:
        scale = rearrange(scale, 'd -> 1 d 1 1')
    l1_distance = torch.abs(gt - scale)
    # l2_distance = torch.pow((gt-scale), 2)
    # exp_distance = torch.exp(torch.abs(gt - scale))
    
    # window_c = l1_distance == l1_distance.min(1, keepdim=True)[0]
    # window = window_c.clone()
    # for r in range(WINDOW_RADIUS):
    #     window[:,1+r:] += window_c[:,:-1-r]
    #     window[:,:-1-r] += window_c[:,1+r:]

    cost = torch.exp(-l1_distance*SCALE_FACTOR)# * window
    soft_label = cost / (cost.sum(1, keepdim=True) + 1e-4)

    return soft_label


def depth_to_stochastic_soft_label(gt, scale, scale_factor=random.randint(10, 30)):
    '''
    convert GT depth map to soft label map
    gt : b 1 h w
    scale : d
    label : b d h w
    '''
    WINDOW_RADIUS = 2
    # SCALE_FACTOR = random.randint(10, 30)
    if len(gt.shape) == 1:
        gt = rearrange(gt, 'n -> 1 n')
        scale = rearrange(scale, 'd -> d 1')
    elif not len(scale.shape) == 4:
        scale = rearrange(scale, 'd -> 1 d 1 1')
    l1_distance = torch.abs(gt - scale)

    cost = torch.exp(-l1_distance*scale_factor)# * window
    soft_label = cost / cost.sum(1, keepdim=True)

    return soft_label


def alex_kendall_reweight_loss(uncertainty, pred_depth, gt):
    loss = (pred_depth - gt).abs()
    loss = loss*torch.exp(-uncertainty) + uncertainty
    return loss.mean()


class AutoLossWeight(nn.Module):
    def __init__(self, *loss_names):
        super().__init__()
        weights = nn.ParameterDict()
        for name in loss_names:
            weights[name] = nn.Parameter(torch.rand(1), requires_grad=True).float()
        self.weights = weights

    def forward(self, **loss_vals):
        assert self.weights.keys() == loss_vals.keys(), f'found contradiction in loss items'
        weighted_loss = 0
        for n,w in self.weights.items():
            weighted_loss += w
            weighted_loss += torch.exp(-w)*loss_vals[n]
        return weighted_loss
