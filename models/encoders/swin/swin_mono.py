import math
import torch
from torch import nn 
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict
from models.encoders.swin.swin_transformer import SwinTransformer

class SwinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinTransformer(
                                    embed_dim=192, 
                                    depths=[2,2,18,2], 
                                    num_heads=[6, 12, 24, 48], 
                                    window_size=7)
        swin_pretrain_dict = torch.load("pretrained/swin_large_patch4_window7_224_22k.pth",
            map_location='cpu')['model']

        self.swin.load_state_dict(swin_pretrain_dict, strict=False)
    
    def forward(self, x):
        features = self.swin(x)
        return features

if __name__ == '__main__':
    model = SwinEncoder()
    x = torch.randn(1, 3, 480, 640)
    y = model(x)
    print([_.shape for _ in y])