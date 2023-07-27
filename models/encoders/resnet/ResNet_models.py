import torch
import torch.nn as nn
import torchvision.models as models
from .ResNet import B2_ResNet

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = B2_ResNet()
        self.initialize_weights()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)       # 64   x 240 x 320
        x = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x)     # 256  x 120 x 160
        x2 = self.resnet.layer2(x1)    # 512  x 60  x 80
        x3 = self.resnet.layer3_1(x2)  # 1024 x 30  x 40
        x4 = self.resnet.layer4_1(x3)  # 2048 x 15  x 20

        return [x0, x1, x2, x3, x4]

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=False)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)