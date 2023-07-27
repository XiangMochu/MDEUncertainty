import torch
import torch.nn as nn
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
import numpy as np

class DenseNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # print('pretrained model not loaded!')
        self.densenet = models.densenet161(pretrained=True)

    def forward(self, x):

        feature_maps = [x]

        for key, value in self.densenet.features._modules.items():
            feature_maps.append(value(feature_maps[-1]))

        return [feature_maps[3],feature_maps[4],feature_maps[6],feature_maps[8],feature_maps[11]]
