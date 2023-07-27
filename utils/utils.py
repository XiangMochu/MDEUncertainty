from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
import math
import requests
import json
import time
import os 
import glob


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)


def compute_depth_errors(gt, pred, errors):
    """Computation of error metrics between predicted and ground truth depths
    """

    loss_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    loss_vals = [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]

    for name, val in zip(loss_names, loss_vals):
        errors[name] = val

    return errors

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )

    return window


def ssim(img1, img2, val_range=10, window_size=11, window=None, size_average=True, full=False):

    L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except Exception as e:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height, width)  # window should be atleast 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability
    C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret


def image_gradients(img, device):

    """works like tf one"""
    if len(img.shape) != 4:
        raise ValueError("Shape mismatch. Needs to be 4 dim tensor")

    img_shape = img.shape
    batch_size, channels, height, width = img.shape

    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]

    shape = np.stack([batch_size, channels, 1, width])
    dy = torch.cat(
        [
            dy,
            torch.zeros(
                [batch_size, channels, 1, width], device=dy.device, dtype=img.dtype
            ),
        ],
        dim=2,
    )
    dy = dy.view(img_shape)

    shape = np.stack([batch_size, channels, height, 1])
    dx = torch.cat(
        [
            dx,
            torch.zeros(
                [batch_size, channels, height, 1], device=dx.device, dtype=img.dtype
            ),
        ],
        dim=3,
    )
    dx = dx.view(img_shape)

    return dy, dx


# Now we define the actual depth loss function
def gradient_criterion(y_true, y_pred, theta=0.1, device="cuda", maxDepth=1000.0 / 10.0):

    # Edges
    dy_true, dx_true = image_gradients(y_true, device)
    dy_pred, dx_pred = image_gradients(y_pred, device)
    l_edges = torch.mean(
        torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), dim=1
    )

    return l_edges

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def send_notice(event_name, key, text):
    url = f"https://maker.ifttt.com/trigger/{event_name}/with/key/{key}"
    payload = {"value1": text}
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
    print(response.text)


def backup_opts(opts, path):
    args = ['-'*10 + str(time.strftime("%Y_%m_%d-%I:%M:%S")) + '-'*10 + '\n']
    for k, v in opts.__dict__.items():
        args.append(f'{k}:{v}\n')
    with open(os.path.join(path, 'args.txt'), 'a') as f:
        f.writelines(args)
    print('Successfully backed up args.')


def backup_code(path):
    target_loc = os.path.join(path, 'code.zip')
    py_files = ' '.join(glob.glob('*.py'))
    folders = ' '.join(['models', 'utils'])
    cmd = f'zip -r {target_loc} {py_files} {folders} > /dev/null'
    os.system(cmd)
    print('Successfully backed up code.')


NAME_DICT = {
    'resnet'      : 'Res', 
    'densenet'    : 'Den', 
    'swin'        : 'Swn', 
    'vit'         : 'Vit', 
    'efficientnet': 'Eff',

    'simple': 'Sim', 
    'bts'   : 'Bts',

    'lin_cls': 'Lin',
    'ada_cls': 'Ada',
    'direct' : 'Dir',

    'none'                 : 'Non',
    'regression_l1'        : 'L-1',
    'regression_rms'       : 'Rms',
    'regression_silog_loss': 'Log',

    'hard_label'            : 'Hrd', 
    'soft_label'            : 'Sft', 
    'soft_label_nll'        : 'Sln',
    'nll'                   : 'Nll',
    'confidence_weighted_l1': 'Cwl',
    'stochastic_soft_label' : 'Ssl',

    'uncertainty_softened_label'       : 'Usl',
    'error_softened_label'             : 'Esl',
    'error_uncertainty_ranking'        : 'Eur',
    'error_uncertainty_ranking_noclamp': 'Eun',
    'error_uncertainty_spearman'       : 'Eus',
    'error_uncertainty_correlation'    : 'Euc',
    'error_uncertainty_l1'             : 'Eul',
    'alex_kendall_reweight'            : 'Akr',

    'mc_dropout': 'Mcd',
    'multi_head': 'Mth',
    'noise'     : 'Noi',
}

def default_name(opts):
    choices = [opts.encoder, opts.decoder, opts.reg_mode, opts.reg_supervision, opts.prob_supervision, opts.uncert_supervision, opts.aleatoric_uncertainty]
    abbr = [NAME_DICT[c] for c in choices]
    if opts.aleatoric_uncertainty == 'none':
        abbr.pop(-1)
        default_name = '_'.join(abbr + [opts.dataset, opts.model_name])
    else:
        default_name = '_'.join(['u'] + abbr + [opts.dataset, opts.model_name])
    return default_name


def unnormalize_image(x):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = rearrange(mean, 'c -> c 1 1')
    std = rearrange(std, 'c -> c 1 1')
    x = (x*std) + mean
    return x