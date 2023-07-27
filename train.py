import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.nyu_data import get_nyuDataset
from utils.kitti_data import get_kittiDataset
from models.endecoder import DepthEstimationModule
from models.losses import RegLoss, ProbLoss, UncertLoss, AutoLossWeight

from utils.options import Options
import utils.options as o
import utils.utils as u
from einops import rearrange
from itertools import chain

n_iter = 0

def main(opts):
    # logging
    os.makedirs(opts.log_path, exist_ok=True)
    opts.model_name = u.default_name(opts)
    writers = {}
    for mode in ['train', 'val']:
        writers[mode] = SummaryWriter(os.path.join(opts.log_path, opts.model_name, mode))
        u.backup_opts(opts, os.path.join(opts.log_path, opts.model_name))
        u.backup_code(os.path.join(opts.log_path, opts.model_name))

    # opts.reg_mode = 'lin_cls'
    # print(f'Enforceing reg_mode as {opts.reg_mode}')

    assert not (opts.prob_supervision == 'none' and opts.reg_supervision == 'none'), 'need supervision'

    opts.max_depth = o.DEFAULTS['max_depth'][opts.dataset]

    generator = DepthEstimationModule(opts.encoder, opts.decoder, opts.reg_mode, 128, 128, (1e-3, opts.max_depth), False)
    
    # gen_state_dict = torch.load('ckpt/0407_lin_res_reg_l1+cwl1/model_epoch1.pth', map_location='cpu')
    # gen_state_dict = {k.replace('module.', ''): v for k, v in gen_state_dict.items()}
    # generator.load_state_dict(gen_state_dict)

    generator.cuda()
    generator_params = generator.parameters()
    
    loss_weight = AutoLossWeight('reg', 'prob', 'uncert').cuda()
    loss_weight_params = loss_weight.parameters()
    optimizer = torch.optim.Adam(chain(generator_params, loss_weight_params), opts.lr_gen)

    # data loading 
    if opts.dataset == 'nyu':
        train_set, test_set = get_nyuDataset(opts.nyu_data_path)
    elif opts.dataset == 'kitti':
        train_set, test_set = get_kittiDataset(opts)

    train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers,
                                pin_memory=True)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)

    for epoch in range(opts.epochs):
        train(opts, train_loader, generator, optimizer, writers, loss_weight)
        val(opts, test_loader, generator, epoch, writers)

        # save
        os.makedirs(os.path.join(opts.save_path, opts.model_name), exist_ok=True)
        torch.save(generator.state_dict(), os.path.join(opts.save_path, opts.model_name, 'model_epoch{}.pth'.format(epoch)))
        print('saved state dict')
        u.adjust_lr(optimizer, opts.lr_gen, epoch, opts.decay_rate, opts.decay_epoch)


def train(opts, train_loader, model, optimizer, writers, loss_weight):
    global n_iter
    batch_time = u.AverageMeter()
    data_time = u.AverageMeter()
    losses = u.AverageMeter(precision=4)

    loss_reg_fn = RegLoss(opts.reg_supervision)
    loss_prob_fn = ProbLoss(opts.prob_supervision)
    loss_uncert_fn = UncertLoss(opts.uncert_supervision)

    end = time.time()

    for i, inputs in enumerate(train_loader):
        tgt_img, tgt_depth = inputs['image'], inputs['depth']
        tgt_orj = tgt_img
        tgt_depth = Variable(tgt_depth).cuda()
        data_time.update(time.time() - end)
        tgt_img = Variable(tgt_img).cuda()

        # if opts.encoder == 'swin':
        #     tgt_img = F.interpolate(tgt_img, size=(384, 384), align_corners=True, mode='bilinear')

        outputs = model(tgt_img)
        pred_depth = F.interpolate(outputs['depth'], size=tgt_depth.shape[-2:], align_corners=True, mode='bilinear')
        
        entropy = outputs['entropy']
        entropy = F.interpolate(entropy, size=tgt_depth.shape[-2:], align_corners=True, mode='bilinear')
        
        uncertainty = outputs['uncert']
        uncertainty = F.interpolate(uncertainty, size=tgt_depth.shape[-2:], align_corners=True, mode='bilinear')

        pred_prob = F.interpolate(outputs['prob'], size=tgt_depth.shape[-2:], align_corners=True, mode='bilinear')
        if opts.reg_mode in ['lin_cls', 'log_cls', 'ada_cls']:
            scales = outputs['scales'].cuda()

        # compute loss        
        weight = tgt_depth > 1e-3 if opts.dataset == 'kitti' else 1
        if opts.dataset == 'kitti' :#and opts.reg_supervision == 'regression_silog_loss':
            mask = tgt_depth.squeeze(1) > 1e-3 
            pred_prob = rearrange(pred_prob, 'b d h w -> d b h w')
            pred_prob = pred_prob[..., mask]
            tgt_depth_ = tgt_depth.squeeze(1)[mask]
            pred_depth_ = pred_depth.squeeze(1)[mask]
            uncertainty_ = uncertainty.squeeze(1)[mask]
            weight = 1
        else:
            tgt_depth_ = tgt_depth
            pred_depth_ = pred_depth
            uncertainty_ = uncertainty
        
        l1_error = torch.abs(pred_depth - tgt_depth) * (tgt_depth > 1e-3).float()
        outputs['l1_error'] = l1_error

        reg_loss = loss_reg_fn(pred_depth=pred_depth_, gt=tgt_depth_, weight=weight)
        prob_loss = loss_prob_fn(pred_prob=pred_prob, scales=scales, gt=tgt_depth_, weight=weight) if opts.prob_supervision != 'none' else 0
        uncert_loss = loss_uncert_fn(uncertainty=uncertainty_, pred_depth=pred_depth_, gt=tgt_depth_, weight=weight) if opts.uncert_supervision != 'none' else 0

        entropy_loss = outputs['entropy'].sum().item() * 1e-4 if 'entropy' in outputs.keys() else 0
        # loss = reg_loss + prob_loss + uncert_loss
        loss = loss_weight(reg=reg_loss, prob=prob_loss, uncert=uncert_loss)

        # compute depth errors, for monitoring training status
        errors = calc_error(opts, pred_depth, tgt_depth)

        losses.update(loss.item(), opts.batch_size)

        errors['loss/reg'] = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
        errors['loss/prob'] = prob_loss.item() if isinstance(prob_loss, torch.Tensor) else prob_loss
        errors['loss/uncert'] = uncert_loss.item() if isinstance(uncert_loss, torch.Tensor) else uncert_loss
        errors['loss/entropy'] = entropy_loss
        errors['loss/total'] = loss.item()

        errors['loss/reg_weight'] = torch.exp(-loss_weight.weights['reg']).item()
        errors['loss/prob_weight'] = torch.exp(-loss_weight.weights['prob']).item()
        errors['loss/uncert_weight'] = torch.exp(-loss_weight.weights['uncert']).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opts.print_freq == 0:
            print('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
            log(opts, writers, 'train', tgt_orj, tgt_depth, outputs, errors)
        
        n_iter += 1

@torch.no_grad()
def val(opts, val_loader, model, epoch, writers):

    model.eval()

    tgt_depth_s = []
    outputs_s = []

    for i, inputs in enumerate(val_loader):
        tgt_img, tgt_depth = inputs['image'], inputs['depth']

        tgt_img = Variable(tgt_img).cuda()
        tgt_depth = Variable(tgt_depth).cuda()
        
        outputs = model(tgt_img)['depth']

        if opts.dataset == 'kitti':
            if inputs['has_valid_depth']:
                tgt_depth = rearrange(tgt_depth, 'b h w c -> b c h w')
                outputs_s.append(outputs.cpu())
                tgt_depth_s.append(tgt_depth.cpu())
        else:
            outputs_s.append(outputs.cpu())
            tgt_depth_s.append(tgt_depth.cpu())
    
    outputs_s = torch.cat(outputs_s)
    tgt_depth_s = torch.cat(tgt_depth_s)
    outputs_s = F.interpolate(outputs_s, tgt_depth_s.shape[-2:], mode='bilinear', align_corners=True)

    errors = calc_error(opts, outputs_s, tgt_depth_s)
    if writers != None:
        log(opts, writers, 'val', None, None, None, errors)

    print('epoch {} '.format(epoch), end='|')
    for name, error in errors.items():
        print(name, ':', error, end=' | ')
    print()


def calc_error(opts, outputs, tgt_depth):
    depth_errors = {}
    pred_depth = torch.clamp(outputs, opts.min_depth, opts.max_depth)
    if opts.dataset == 'kitti':
        depth_errors = u.compute_depth_errors(tgt_depth[tgt_depth>1e-3], pred_depth[tgt_depth>1e-3], depth_errors)
    else:
        depth_errors = u.compute_depth_errors(tgt_depth, pred_depth, depth_errors)
    return depth_errors


def log(opts, writers, mode, img, gt, outputs, errors):
    global n_iter

    writer = writers[mode]

    for l, v in errors.items():
        writer.add_scalar('{}'.format(l), v, n_iter)
    
    if mode == 'train':
        pred_depth = outputs['depth']
        l1_error = outputs['l1_error']
        uncertainty = outputs['uncert']

        for i in range(min(4, opts.batch_size)): # frames
            writer.add_image("color/{}".format(i), u.unnormalize_image(img[i].data), n_iter)
            writer.add_image("gt/{}".format(i), u.normalize_image(gt[i].data), n_iter)
            writer.add_image("pred/{}".format(i),
                u.normalize_image(pred_depth[i].data), n_iter)
            writer.add_image("l1_error/{}".format(i),
                u.normalize_image(l1_error[i].data), n_iter)
            writer.add_image("uncertainty/{}".format(i),
                u.normalize_image(uncertainty[i].data), n_iter)

options = Options()
opts = options.parse()

if __name__ == '__main__':
    if opts.phone_notify:
        try:
            main(opts)
        except Exception as e:
            print(e)
            u.send_notice('notice_phone', opts.ifttt_key, '')
    else:
        main(opts)