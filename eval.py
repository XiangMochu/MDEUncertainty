import os 
import torch 
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.nyu_data import get_nyuDataset
from utils.kitti_data import get_kittiDataset
from models.endecoder import DepthEstimationModule
from models.endecoder import MCDropoutDepthEstimationModule, MultiHeadDepthEstimationModule, NoiseInjectedDepthEstimationModule

from einops import rearrange
from tqdm import tqdm

import pickle

from scipy.stats import spearmanr

import numpy as np 
import sys


UNCERTAINTY_MODELS = {
    'mc_dropout': MCDropoutDepthEstimationModule, 
    'multi_head': MultiHeadDepthEstimationModule, 
    'noise': NoiseInjectedDepthEstimationModule, 
    'predictive': None,
}

class Config:
    alea = 'none'
    nyu_data_path = 'MODIFY_ME/nyu_data.zip'
    encoder = 'resnet'
    decoder = 'simple'
    reg_mode = 'lin_cls'
    ckpt = 'model_epoch9.pth'

def parse_config(path, subfolder=None):
    cfg = Config()
    cfg.ckpt = os.path.join('ckpt', subfolder, path, 'model_epoch9.pth') if subfolder is not None else os.path.join('ckpt', path, 'model_epoch9.pth') 
    cfgs = path.split('_')
    if path.startswith('u_'):
        cfg.alea = {'Mcd':'mc_dropout', 'Mth':'multi_head', 'Noi':'noise'}[cfgs[7]]
        offset = 1
    else:
        offset = 0
    cfg.encoder = {'Den':'densenet', 'Eff':'efficientnet', 'Res':'resnet', 'Swn':'swin', 'Vit':'vit'}[cfgs[0+offset]]
    cfg.decoder = {'Sim':'simple', 'Bts':'bts'}[cfgs[1+offset]]
    cfg.reg_mode = {'Lin':'lin_cls', 'Ada':'ada_cls', 'Dir':'direct'}[cfgs[2+offset]]

    return cfg

uncertainty_metrics = ["abs_rel", "rmse", "a1"]
def compute_eigen_errors_v2(gt, pred, metrics=uncertainty_metrics, mask=None, reduce_mean=False):
    """Revised compute_eigen_errors function used for uncertainty metrics, with optional reduce_mean argument and (1-a1) computation
    """
    results = []
    
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]
    
    if "abs_rel" in metrics:
        abs_rel = (np.abs(gt - pred) / gt)
        if reduce_mean:
            abs_rel = abs_rel.mean()
        results.append(abs_rel)

    if "rmse" in metrics:
        rmse = (gt - pred) ** 2
        if reduce_mean:
            rmse = np.sqrt(rmse.mean())
        results.append(rmse)

    if "a1" in metrics:
        a1 = np.maximum((gt / pred), (pred / gt))
        if reduce_mean:
        
            # invert to get outliers
            a1 = (a1 >= 1.25).mean()
        results.append(a1)

    return results

def compute_aucs(gt, pred, uncert, intervals=50):
    """Computation of auc metrics
    """
    
    # results dictionaries
    AUSE = {"abs_rel":0, "rmse":0, "a1":0}
    AURG = {"abs_rel":0, "rmse":0, "a1":0}

    # revert order (high uncertainty first)
    uncert = -uncert
    true_uncert = compute_eigen_errors_v2(gt,pred)
    true_uncert = {"abs_rel":-true_uncert[0],"rmse":-true_uncert[1],"a1":-true_uncert[2]}

    # prepare subsets for sampling and for area computation
    quants = [100./intervals*t for t in range(0,intervals)]
    plotx = [1./intervals*t for t in range(0,intervals+1)]

    # get percentiles for sampling and corresponding subsets
    thresholds = [np.percentile(uncert, q) for q in quants]
    subs = [(uncert >= t) for t in thresholds]

    # compute sparsification curves for each metric (add 0 for final sampling)
    sparse_curve = {m:[compute_eigen_errors_v2(gt,pred,metrics=[m],mask=sub,reduce_mean=True)[0] for sub in subs]+[0] for m in uncertainty_metrics }

    # human-readable call
    '''
    sparse_curve =  {"rmse":[compute_eigen_errors_v2(gt,pred,metrics=["rmse"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0], 
                     "a1":[compute_eigen_errors_v2(gt,pred,metrics=["a1"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0],
                     "abs_rel":[compute_eigen_errors_v2(gt,pred,metrics=["abs_rel"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0]}
    '''
    
    # get percentiles for optimal sampling and corresponding subsets
    opt_thresholds = {m:[np.percentile(true_uncert[m], q) for q in quants] for m in uncertainty_metrics}
    opt_subs = {m:[(true_uncert[m] >= o) for o in opt_thresholds[m]] for m in uncertainty_metrics}

    # compute sparsification curves for optimal sampling (add 0 for final sampling)
    opt_curve = {m:[compute_eigen_errors_v2(gt,pred,metrics=[m],mask=opt_sub,reduce_mean=True)[0] for opt_sub in opt_subs[m]]+[0] for m in uncertainty_metrics}

    # compute metrics for random sampling (equal for each sampling)
    rnd_curve = {m:[compute_eigen_errors_v2(gt,pred,metrics=[m],mask=None,reduce_mean=True)[0] for t in range(intervals+1)] for m in uncertainty_metrics}    

    # compute error and gain metrics
    for m in uncertainty_metrics:

        # error: subtract from method sparsification (first term) the oracle sparsification (second term)
        AUSE[m] = np.trapz(sparse_curve[m], x=plotx) - np.trapz(opt_curve[m], x=plotx)
        
        # gain: subtract from random sparsification (first term) the method sparsification (second term)
        AURG[m] = rnd_curve[m][0] - np.trapz(sparse_curve[m], x=plotx)

    # returns a dictionary with AUSE and AURG for each metric
    return {m:[AUSE[m], AURG[m]] for m in uncertainty_metrics}


def get_curve(known, novel):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def compute_fpr95_auroc(uncert, a1_map):
    right_uncert = uncert[a1_map]
    wrong_uncert = uncert[~a1_map]

    tp, fp, fpr_at_tpr95 = get_curve(wrong_uncert, right_uncert)

    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    auroc = -np.trapz(1.-fpr, tpr)

    return fpr_at_tpr95, auroc


class Metrics:
    def __init__(self):
        # accuracy
        self.absrel = []
        self.log10 = []
        self.rmse = []
        self.sqrel = []
        self.logrms = []
        self.a1 = []
        self.a2 = []
        self.a3 = []
        
        # uncertainty
        self.ause_rmse_e = []
        self.aurg_rmse_e = []
        self.ause_absrel_e = []
        self.aurg_absrel_e = []
        self.ause_a1_e = []
        self.aurg_a1_e = []
        self.spearman_e = []
        self.fpr95_e = []
        self.auroc_e = []

        # only for classification
        self.ause_rmse_v = []
        self.aurg_rmse_v = []
        self.ause_absrel_v = []
        self.aurg_absrel_v = []
        self.ause_a1_v = []
        self.aurg_a1_v = []
        self.spearman_v = []
        self.fpr95_v = []
        self.auroc_v = []

        self.ause_rmse_ve = []
        self.aurg_rmse_ve = []
        self.ause_absrel_ve = []
        self.aurg_absrel_ve = []
        self.ause_a1_ve = []
        self.aurg_a1_ve = []
        self.spearman_ve = []
        self.fpr95_ve = []
        self.auroc_ve = []
        
    def update(self, prob, scale, depth_pred, gt, uncert, uncert_as_prob=False):
        ### measure accuracy
        thresh = np.maximum((gt / depth_pred), (depth_pred / gt))
        a1_map = thresh < 1.25
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rms = (gt - depth_pred) ** 2
        rms = np.sqrt(rms.mean())

        log_rms = (np.log(gt) - np.log(depth_pred)) ** 2
        log_rms = np.sqrt(log_rms.mean())

        abs_rel = np.mean(np.abs(gt - depth_pred) / gt)
        sq_rel = np.mean(((gt - depth_pred) ** 2) / gt)

        err = np.abs(np.log10(depth_pred) - np.log10(gt))
        log10 = np.mean(err)

        self.a1.append(a1)
        self.a2.append(a2)
        self.a3.append(a3)
        self.rmse.append(rms)
        self.logrms.append(log_rms)
        self.absrel.append(abs_rel)
        self.sqrel.append(sq_rel)
        self.log10.append(log10)
        ###

        if uncert_as_prob:
            uncert_e = prob
        else:
            uncert_e = (- prob * np.log(prob + 1e-8)).sum(1, keepdims=True)
        aucs_e = compute_aucs(gt, depth_pred, uncert_e)
        spearman_e = spearmanr(uncert_e.flatten(), np.abs(gt-depth_pred).flatten())[0]
        
        self.ause_rmse_e.append(float(aucs_e['rmse'][0]))
        self.aurg_rmse_e.append(float(aucs_e['rmse'][1]))
        self.ause_absrel_e.append(float(aucs_e['abs_rel'][0]))
        self.aurg_absrel_e.append(float(aucs_e['abs_rel'][1]))
        self.ause_a1_e.append(float(aucs_e['a1'][0]))
        self.aurg_a1_e.append(float(aucs_e['a1'][1]))
        self.spearman_e.append(float(spearman_e))

        if not (np.all(a1_map) or np.all(~a1_map)):
            fpr95_e, auroc_e = compute_fpr95_auroc(uncert_e, a1_map)
            self.fpr95_e.append(float(fpr95_e))
            self.auroc_e.append(float(auroc_e))
        else:
            print('skipping evaluating fpr95 and auroc on this sample')

        if scale is not None:
            prob_errors = np.abs((scale - depth_pred))
            uncert_v = (prob * prob_errors).sum(1, keepdims=True)
            aucs_v = compute_aucs(gt, depth_pred, uncert_v)
            spearman_v = spearmanr(uncert_v.flatten(), np.abs(gt-depth_pred).flatten())[0]

            self.ause_rmse_v.append(float(aucs_v['rmse'][0]))
            self.aurg_rmse_v.append(float(aucs_v['rmse'][1]))
            self.ause_absrel_v.append(float(aucs_v['abs_rel'][0]))
            self.aurg_absrel_v.append(float(aucs_v['abs_rel'][1]))
            self.ause_a1_v.append(float(aucs_v['a1'][0]))
            self.aurg_a1_v.append(float(aucs_v['a1'][1]))
            self.spearman_v.append(float(spearman_v))

            if not (np.all(a1_map) or np.all(~a1_map)):
                fpr95_v, auroc_v = compute_fpr95_auroc(uncert_v, a1_map)
                self.fpr95_v.append(float(fpr95_v))
                self.auroc_v.append(float(auroc_v))

            uncert_ve = uncert
            aucs_ve = compute_aucs(gt, depth_pred, uncert_ve)
            spearman_ve = spearmanr(uncert_ve.flatten(), np.abs(gt-depth_pred).flatten())[0]
            self.ause_rmse_ve.append(float(aucs_ve['rmse'][0]))
            self.aurg_rmse_ve.append(float(aucs_ve['rmse'][1]))
            self.ause_absrel_ve.append(float(aucs_ve['abs_rel'][0]))
            self.aurg_absrel_ve.append(float(aucs_ve['abs_rel'][1]))
            self.ause_a1_ve.append(float(aucs_ve['a1'][0]))
            self.aurg_a1_ve.append(float(aucs_ve['a1'][1]))
            self.spearman_ve.append(float(spearman_ve))

            if not (np.all(a1_map) or np.all(~a1_map)):
                fpr95_ve, auroc_ve = compute_fpr95_auroc(uncert_ve, a1_map)
                self.fpr95_ve.append(float(fpr95_ve))
                self.auroc_ve.append(float(auroc_ve))
        ###
    
    def reduce(self, metric, method):
        if method == 'mean':
            return metric.mean()
        else:
            return metric

    def get_metrics(self, reduce='mean', is_prob=False):
        absrel    = self.reduce(torch.tensor(self.absrel), reduce)
        log10     = self.reduce(torch.tensor(self.log10), reduce)
        rmse      = self.reduce(torch.tensor(self.rmse), reduce)
        sqrel     = self.reduce(torch.tensor(self.sqrel), reduce)
        logrms    = self.reduce(torch.tensor(self.logrms), reduce)
        a1        = self.reduce(torch.tensor(self.a1), reduce)
        a2        = self.reduce(torch.tensor(self.a2), reduce)
        a3        = self.reduce(torch.tensor(self.a3), reduce)

        ause_rmse_e = self.reduce(torch.tensor(self.ause_rmse_e), reduce)
        aurg_rmse_e = self.reduce(torch.tensor(self.aurg_rmse_e), reduce)
        ause_absrel_e = self.reduce(torch.tensor(self.ause_absrel_e), reduce)
        aurg_absrel_e = self.reduce(torch.tensor(self.aurg_absrel_e), reduce)
        ause_a1_e = self.reduce(torch.tensor(self.ause_a1_e), reduce)
        aurg_a1_e = self.reduce(torch.tensor(self.aurg_a1_e), reduce)
        spearman_e = self.reduce(torch.tensor(self.spearman_e), reduce)

        fpr95_e = self.reduce(torch.tensor(self.fpr95_e), reduce)
        auroc_e = self.reduce(torch.tensor(self.auroc_e), reduce)

        if is_prob:
            ause_rmse_v = self.reduce(torch.tensor(self.ause_rmse_v), reduce)
            aurg_rmse_v = self.reduce(torch.tensor(self.aurg_rmse_v), reduce)
            ause_absrel_v = self.reduce(torch.tensor(self.ause_absrel_v), reduce)
            aurg_absrel_v = self.reduce(torch.tensor(self.aurg_absrel_v), reduce)
            ause_a1_v = self.reduce(torch.tensor(self.ause_a1_v), reduce)
            aurg_a1_v = self.reduce(torch.tensor(self.aurg_a1_v), reduce)
            spearman_v = self.reduce(torch.tensor(self.spearman_v), reduce)

            fpr95_v = self.reduce(torch.tensor(self.fpr95_v), reduce)
            auroc_v = self.reduce(torch.tensor(self.auroc_v), reduce)

            ause_rmse_ve = self.reduce(torch.tensor(self.ause_rmse_ve), reduce)
            aurg_rmse_ve = self.reduce(torch.tensor(self.aurg_rmse_ve), reduce)
            ause_absrel_ve = self.reduce(torch.tensor(self.ause_absrel_ve), reduce)
            aurg_absrel_ve = self.reduce(torch.tensor(self.aurg_absrel_ve), reduce)
            ause_a1_ve = self.reduce(torch.tensor(self.ause_a1_ve), reduce)
            aurg_a1_ve = self.reduce(torch.tensor(self.aurg_a1_ve), reduce)
            spearman_ve = self.reduce(torch.tensor(self.spearman_ve), reduce)

            fpr95_ve = self.reduce(torch.tensor(self.fpr95_ve), reduce)
            auroc_ve = self.reduce(torch.tensor(self.auroc_ve), reduce)

            return dict(
                absrel=absrel, log10=log10, rmse=rmse, sqrel=sqrel, logrms=logrms, a1=a1, a2=a2, a3=a3,
                ause_rmse_e=ause_rmse_e, aurg_rmse_e=aurg_rmse_e, ause_absrel_e=ause_absrel_e, aurg_absrel_e=aurg_absrel_e, 
                ause_a1_e=ause_a1_e, aurg_a1_e=aurg_a1_e, fpr95_e=fpr95_e, auroc_e=auroc_e, spearman_e=spearman_e,
                ause_rmse_v=ause_rmse_v, aurg_rmse_v=aurg_rmse_v, ause_absrel_v=ause_absrel_v, aurg_absrel_v=aurg_absrel_v, 
                ause_a1_v=ause_a1_v, aurg_a1_v=aurg_a1_v, fpr95_v=fpr95_v, auroc_v=auroc_v, spearman_v=spearman_v,
                ause_rmse_ve=ause_rmse_ve, aurg_rmse_ve=aurg_rmse_ve, ause_absrel_ve=ause_absrel_ve, aurg_absrel_ve=aurg_absrel_ve,
                ause_a1_ve=ause_a1_ve, aurg_a1_ve=aurg_a1_ve, fpr95_ve=fpr95_ve, auroc_ve=auroc_ve, spearman_ve=spearman_ve)
        else:
            return dict(
                absrel=absrel, log10=log10, rmse=rmse, sqrel=sqrel, logrms=logrms, a1=a1, a2=a2, a3=a3,
                ause_rmse_e=ause_rmse_e, aurg_rmse_e=aurg_rmse_e, ause_absrel_e=ause_absrel_e, aurg_absrel_e=aurg_absrel_e, 
                ause_a1_e=ause_a1_e, aurg_a1_e=aurg_a1_e, fpr95_e=fpr95_e, auroc_e=auroc_e, spearman_e=spearman_e)


@torch.no_grad()
def eval(cfg):
    max_depth = 10 if 'nyu' in cfg.ckpt else 80
    if cfg.alea == 'none':
        generator = DepthEstimationModule(cfg.encoder, cfg.decoder, cfg.reg_mode, 128, 128, (1e-3, max_depth), False)
    else:
        if cfg.alea in ['mc_dropout', 'noise']:
            generator = UNCERTAINTY_MODELS[cfg.alea](cfg.encoder, cfg.decoder, cfg.reg_mode, 128, 128, (1e-3, max_depth), False, fw_pass=3)
        else:
            generator = UNCERTAINTY_MODELS[cfg.alea](cfg.encoder, cfg.decoder, cfg.reg_mode, 128, 128, (1e-3, max_depth), False)
    
    state_dict = torch.load(cfg.ckpt, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if "regressor.alpha2" in state_dict.keys():
        state_dict.__delitem__("regressor.alpha2")
    if "regressors.0.alpha2" in state_dict.keys():
        state_dict.__delitem__("regressors.0.alpha2")
        state_dict.__delitem__("regressors.1.alpha2")
        state_dict.__delitem__("regressors.2.alpha2")
    generator.load_state_dict(state_dict)
    generator.cuda()
    if not cfg.alea == 'mc_dropout':
        generator.eval()

    if 'nyu' in cfg.ckpt:
        _, test_set = get_nyuDataset(cfg.nyu_data_path)
    elif 'kitti' in cfg.ckpt:
        _, test_set = get_kittiDataset(cfg)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    metrics = Metrics()
    if cfg.alea != 'none':
        metrics_var = Metrics()

    if cfg.reg_mode == 'lin_cls':
        scales = generator.scales.cpu()
        scales = rearrange(scales, 'd -> 1 d 1') if 'kitti' in cfg.ckpt else rearrange(scales, 'd -> 1 d 1 1')

    idx = 0
    for inputs in tqdm(test_loader):
        tgt_img, tgt_depth = inputs['image'], inputs['depth']
        if 'kitti' in cfg.ckpt:
            if inputs['has_valid_depth'] == False:
                continue
            tgt_depth = rearrange(tgt_depth, 'b h w c -> b c h w')

        tgt_img = tgt_img.cuda()

        outputs = generator(tgt_img)

        if cfg.alea == 'none':
            prob = F.interpolate(outputs['prob'], size=(tgt_depth.shape[2], tgt_depth.shape[3]), mode='bilinear').cpu()
            pred_depth = F.interpolate(outputs['depth'], size=(tgt_depth.shape[2], tgt_depth.shape[3]), mode='bilinear').cpu()
            uncertainty = F.interpolate(outputs['uncert'], size=(tgt_depth.shape[2], tgt_depth.shape[3]), mode='bilinear').cpu()

            if cfg.reg_mode == 'ada_cls':
                scales = outputs['scales'].cpu()
        else:
            prob = F.interpolate(outputs['prob'].mean(0), size=(tgt_depth.shape[2], tgt_depth.shape[3]), mode='bilinear').cpu()
            pred_depth = F.interpolate(outputs['depth'].mean(0), size=(tgt_depth.shape[2], tgt_depth.shape[3]), mode='bilinear').cpu()
            uncertainty = F.interpolate(outputs['uncert'].mean(0), size=(tgt_depth.shape[2], tgt_depth.shape[3]), mode='bilinear').cpu()

            uncertainty_var = F.interpolate(outputs['depth'].var(0), size=(tgt_depth.shape[2], tgt_depth.shape[3]), mode='bilinear').cpu()

            if cfg.reg_mode == 'ada_cls':
                scales = outputs['scales'].cpu().mean(0)
        
        
        
        # do garg crop
        if 'kitti' in cfg.ckpt:
            gt_height, gt_width = tgt_depth.shape[-2:]
            eval_mask = torch.zeros(gt_height, gt_width)
            eval_mask[..., int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            eval_mask[tgt_depth[0,0]<1e-3] = 0
            eval_mask[tgt_depth[0,0]>80] = 0
            prob = prob[..., eval_mask == 1]
            pred_depth = pred_depth[..., eval_mask == 1]
            tgt_depth = tgt_depth[..., eval_mask == 1]
            uncertainty = uncertainty[..., eval_mask == 1]

        idx += 1

        if cfg.reg_mode == 'ada_cls' and 'kitti' in cfg.ckpt:
            prob = prob.unsqueeze(-1)
            pred_depth = pred_depth.unsqueeze(-1)
            tgt_depth = tgt_depth.unsqueeze(-1)
            uncertainty = uncertainty.unsqueeze(-1)

        scale_ = None
        metrics.update(prob=prob.cpu().numpy(), scale=scale_, depth_pred=pred_depth.cpu().numpy(), gt=tgt_depth.cpu().numpy(), uncert=uncertainty.cpu().numpy())
        
        if cfg.alea != 'none':
            if 'kitti' in cfg.ckpt:
                uncertainty_var = uncertainty_var[..., eval_mask == 1]
            metrics_var.update(prob=uncertainty_var.cpu().numpy(), scale=None, depth_pred=pred_depth.cpu().numpy(), gt=tgt_depth.cpu().numpy(), uncert=None, uncert_as_prob=True)
 
    metrics_dict = metrics.get_metrics(reduce='mean', is_prob=(cfg.reg_mode != 'direct'))
    print(' | '.join(['{}: {:.4f}'.format(k, v) for k, v in metrics_dict.items()]))

    pkl_name = cfg.ckpt.split('/')[-2].split('.')[0]
    pickle.dump(metrics.__dict__, open(f'metrics/{pkl_name}.pkl', 'wb'))

    if cfg.alea != 'none':
        metrics_dict_var = metrics_var.get_metrics(reduce='mean', is_prob=False)
        print('Uncertainty as variance of output:')
        print(' | '.join(['{}: {:.4f}'.format(k, v) for k, v in metrics_dict_var.items()]))
        pickle.dump(metrics_var.__dict__, open(f'metrics/{pkl_name}_var.pkl', 'wb'))
    

def get_uncertainty_map(pred_dict):
    scales = pred_dict['scales']
    pred_probs = pred_dict['pred_prob']
    gts = pred_dict['tgt_depth']

    cwl1_weights = []
    nll_weights = []

    scales = rearrange(scales, 'd -> 1 d 1 1')
    for a_prob, a_gt in zip(pred_probs, gts):
        errors = torch.abs((scales - a_gt))
        weighted = a_prob * errors
        cwl1_weights.append(weighted)
        
        nll = - a_prob * torch.log(a_prob + 1e-8)
        nll_weights.append(nll)
    
    cwl1_weights = torch.cat(cwl1_weights)
    nll_weights = torch.cat(nll_weights)
    return {'cwl1': cwl1_weights, 'nll':nll_weights}


def get_errors(pred_dict):
    pred_depths = pred_dict['pred_depth']
    gts = pred_dict['tgt_depth']

    errors = gts - pred_depths
    thresh = torch.max((gts / pred_depths), (pred_depths / gts))

    a1 = (thresh < 1.25).float().mean([1,2,3])
    rmse = torch.sqrt(errors ** 2).mean([1,2,3])
    abs_rel = (torch.abs(gts - pred_depths) / gts).mean([1,2,3])

    return {'res_error': errors, 'a1': a1, 'rmse': rmse, 'abs_rel': abs_rel}
    

if __name__ == '__main__':
    # cfg = parse_config('Res_Sim_Lin_L-1_Non_kitti_2022_05_11-11:21:29', subfolder='_backbones')
    # path = 'Res_Sim_Lin_L-1_Non_Eur_kitti_2023_02_28-22:42:51'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    print('EVALUATING:', path)
    cfg = parse_config(path, subfolder=None)
    eval(cfg)
