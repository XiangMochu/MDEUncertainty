from logging import exception
import math
import numpy as np
import torch 
from torch import nn 
from torch.nn import functional as F

from models.encoders.resnet.ResNet_models import ResNetEncoder
from models.encoders.densenet.DenseNet_models import DenseNetEncoder
from models.encoders.swin.swin_mono import SwinEncoder
from models.encoders.vit.vit_mono import ViTEncoder
from models.encoders.efficientnet.EfficientNet_models import EfficientNetEncoder

from models.decoders import SimpleDecoder, BTSDecoder

from models.regressors import RegressionHead, ClassificationHead, AdaptiveClassificationHead


ENCODER = {
    'resnet':      ResNetEncoder,   # resnet50
    'densenet':    DenseNetEncoder, # densenet161
    'swin':        SwinEncoder,     # swin_base_patch4_window12_384
    'vit':         ViTEncoder,      # vitb_rn50_384
    'efficientnet':EfficientNetEncoder}

ENC_FEAT = { # feature channel, downsample factor
    'resnet':      [[ 64,2],[256,4],[512, 8],[1024,16],[2048,32]],
    'densenet':    [[ 96,2],[ 96,4],[192, 8],[ 384,16],[2208,32]],
    # 'swin':        [[128,4],[256,8],[512,16],[1024,32],[1024,32]],
    'swin':        [[192,4],[384,8],[768,16],[1536,32]],
    'vit':         [[256,4],[512,8],[768,16],[ 768,32]],
    'efficientnet':[[ 24,2],[ 40,4],[ 64, 8],[ 176,16],[2048,32]]}

DECODER = {
    'simple': SimpleDecoder,
    'bts':    BTSDecoder
}


class DepthEstimationModule(nn.Module):
    def __init__(self, enc_mode, dec_mode, reg_mode,
                 decoder_feat, num_class=None, depth_range=None, hard_pred=None):
        super().__init__()
        assert enc_mode in ['resnet', 'densenet', 'swin', 'vit', 'efficientnet'], \
            f'encoder "{enc_mode} not available.'
        self.encoder = ENCODER[enc_mode]()
        self.enc_feet = torch.tensor(ENC_FEAT[enc_mode])[:, 0]
        self.dec_feat = decoder_feat
        
        self.decoder = DECODER[dec_mode](self.enc_feet, self.dec_feat, enc_mode, depth_range)
        
        if reg_mode in ['inverse', 'direct']:
            # assert num_class is None, 'unnecessary argument'
            # assert depth_range is None, 'unnecessary argument'
            self.regressor = RegressionHead(self.dec_feat, num_class)

        elif reg_mode in ['lin_cls', 'log_cls']:
            assert num_class is not None, 'necessary argument missing'
            assert depth_range is not None, 'necessary argument missing'
            assert hard_pred is not None, 'necessary argument missing'

            scales = self.get_class_scales(reg_mode, num_class, depth_range)
            self.register_buffer('scales', scales)
            self.regressor = ClassificationHead(self.dec_feat, num_class, scales, hard_pred)
        
        elif reg_mode in ['ada_cls']:
            self.regressor = AdaptiveClassificationHead(self.dec_feat, num_class, *depth_range)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)\
            if enc_mode in ['swin', 'vit'] else nn.Identity()
    
    def get_class_scales(self, reg_mode, num_class, depth_range):
        min_depth, max_depth = depth_range
        if 'log' in reg_mode:
            scales = torch.exp(torch.linspace(math.log(min_depth), math.log(max_depth), num_class+1))
        elif 'lin' in reg_mode:
            scales = torch.linspace(min_depth, max_depth, num_class+1)
        scales = (scales[1:]+scales[:-1])/2
        return scales
    
    def forward(self, x, temperature=1.0, vis_act=False):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        if vis_act:
            act = self.regressor.get_penultimate(x)
        x = self.regressor(x, temperature)
        if vis_act:
            return x, act
        return x


def dict_stack(xs, exception='scales'):
    xs_stack = {}
    for k in xs[0].keys():
        if k == exception:
            xs_stack[k] = xs[0][k]
        else:
            xs_stack[k] = torch.stack([x[k] for x in xs])#.mean(dim=0)
    return xs_stack


class MultiHeadDepthEstimationModule(DepthEstimationModule):
    def __init__(self, enc_mode, dec_mode, reg_mode, decoder_feat,
        num_class=None, depth_range=None, hard_pred=None, head_num=3):
        super().__init__(enc_mode, dec_mode, reg_mode,
                         decoder_feat, num_class, depth_range, hard_pred)
        
        self.head_num = head_num
        del self.decoder
        del self.regressor
        self.decoders = nn.ModuleDict()
        self.regressors = nn.ModuleDict()

        for i in range(head_num):
            self.decoders[str(i)] = DECODER[dec_mode](self.enc_feet, self.dec_feat, enc_mode, depth_range)

            if reg_mode in ['inverse', 'direct']:
                self.regressors[str(i)] = RegressionHead(self.dec_feat)
            elif reg_mode in ['lin_cls', 'log_cls', 'lin_car', 'log_car']:
                self.regressors[str(i)] = ClassificationHead(self.dec_feat, num_class, self.scales, hard_pred)
    
    def forward(self, x):
        xs = []
        x_ = self.encoder(x)
        for i in range(self.head_num):
            x = self.decoders[str(i)](x_)
            x = self.regressors[str(i)](x)
            xs.append(x)
        return dict_stack(xs, exception='scales')


class MCDropoutDepthEstimationModule(DepthEstimationModule):
    def __init__(self, enc_mode, dec_mode, reg_mode, decoder_feat, 
        num_class=None, depth_range=None, hard_pred=None, drop_rate=0.3, fw_pass=3):
        super().__init__(enc_mode, dec_mode, reg_mode, 
            decoder_feat, num_class, depth_range, hard_pred)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(self.drop_rate)
        self.fw_pass = fw_pass
    
    def forward(self, x_input):
        xs = []
        for i in range(self.fw_pass):
            x = self.encoder(x_input)
            x = [self.dropout(_x) for _x in x]
            x = self.decoder(x)
            x = self.regressor(x)
            xs.append(x)
            del x
        return dict_stack(xs, exception='scales')


class NoiseInjectedDepthEstimationModule(DepthEstimationModule):
    def __init__(self, enc_mode, dec_mode, reg_mode, decoder_feat, 
            num_class=None, depth_range=None, hard_pred=None, latent_dim=24, fw_pass=1):
        super().__init__(enc_mode, dec_mode, reg_mode, decoder_feat, 
            num_class, depth_range, hard_pred)
    
        self.latent_dim = latent_dim
        ch = self.enc_feet[-1]
        self.noise_conv = nn.Conv2d(ch + latent_dim, ch, kernel_size=3, padding=1)
        self.spatial_axes = [2, 3]
        self.fw_pass = fw_pass

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            a.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, x_img):
        xs = []
        x_ = self.encoder(x_img)

        x_last = x_[-1].clone()
        for i in range(self.fw_pass):
            z = torch.randn(x_img.shape[0], self.latent_dim).to(x_img.device)

            z_noise = torch.unsqueeze(z, 2)
            z_noise = self.tile(z_noise, 2, x_last.shape[self.spatial_axes[0]])
            z_noise = torch.unsqueeze(z_noise, 3)
            z_noise = self.tile(z_noise, 3, x_last.shape[self.spatial_axes[1]])
            x_last_ = torch.cat((x_last, z_noise), 1)
            x_last_ = self.noise_conv(x_last_)
            
            x_[-1] = x_last_

            x = self.decoder(x_)
            x = self.regressor(x)

            xs.append(x)
        
        return dict_stack(xs, exception='scales')


if __name__ == '__main__':
    from itertools import product
    for (enc_mode, dec_mode, reg_mode) in product(ENCODER, DECODER, ['lin_cls', 'log_cls', 'ada_cls']):
        print(enc_mode, dec_mode, reg_mode)
    # mde = DepthEstimationModule('swin', 'simple', 'inverse', 256)#, 128, (1e-3, 10))
    # x = torch.randn(1, 3, 384, 384)
    # y = mde(x)
    # print(y['depth'].shape)