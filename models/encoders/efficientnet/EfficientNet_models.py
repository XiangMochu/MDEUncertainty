import torch 
from torch import nn

class EfficientNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_model = self._efficient_basemodel()
    
    def _efficient_basemodel(self):
        basemodel_name = 'tf_efficientnet_b5_ap'

        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        return basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features[4], features[5], features[6], features[8], features[11]

if __name__ == '__main__':
    effi_enc = EfficientNetEncoder()
    x = torch.randn(1, 3, 224, 224)
    fs = effi_enc(x)
    print([f.shape for f in fs])