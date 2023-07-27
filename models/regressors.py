import torch 
from torch import mv, nn 
from torch.nn import functional as F
from einops import rearrange

class RegressionHead(nn.Module):
    def __init__(self, feat, num_class):
        super().__init__()
        self.depth_head1 = nn.Sequential(
            nn.BatchNorm2d(feat),
            nn.LeakyReLU(),
            nn.Conv2d(feat, num_class, kernel_size=3, padding=1))
        
        self.depth_head2 = nn.Conv2d(num_class, 1, kernel_size=1)

        self.alpha1 = nn.Parameter(torch.ones(1))
    
    def get_penultimate(self, x):
        return F.softmax(self.depth_head1(x), dim=1)
    
    def forward(self, x, temp=None):
        feat1 = self.depth_head1(x)
        depth = self.depth_head2(feat1)
        prob = F.softmax(feat1, dim=1)
        entropy = (-prob * torch.log(torch.clamp(prob, 1e-4, 1.0-1e-4))).sum(1, True)
        uncert = entropy * F.softplus(self.alpha1)
        return {'depth': depth, 'prob': prob, 'entropy': entropy, 'uncert': uncert}


class ClassificationHead(nn.Module):
    def __init__(self, feat, num_class, scales, hard_pred):
        super().__init__()
        self.depth_head = nn.Sequential(
            nn.BatchNorm2d(feat),
            nn.LeakyReLU(),
            nn.Conv2d(feat, num_class, kernel_size=3, stride=1, padding=1),
            # nn.Softmax(dim=1)
            )

        self.scales = scales
        scales_ = rearrange(self.scales, 'd -> 1 d 1 1')
        self.hard_pred = hard_pred
        self.register_buffer('scales_', scales_)

        self.alpha1 = nn.Parameter(torch.ones(1))
        # self.alpha2 = nn.Parameter(torch.ones(1))
    
    def get_penultimate(self, x):
        return F.softmax(self.depth_head(x), dim=1)
    
    def forward(self, x, temp=1):
        prob = self.depth_head(x)
        prob = F.softmax(prob * temp, dim=1)
        if self.hard_pred:
            max_prob = (prob == prob.max(dim=1, keepdim=True)[0]).to(torch.float)
            depth = torch.einsum('bdhw,d->bhw', max_prob, self.scales.to(prob.device)).unsqueeze(1)
        else:
            depth = torch.einsum('bdhw,d->bhw', prob, self.scales.to(prob.device)).unsqueeze(1)
        
        entropy = (-prob * torch.log(torch.clamp(prob, 1e-4, 1.0-1e-4))).sum(1, True)
        # return {'prob':prob, 'depth':depth, 'entropy':entropy}
        
        # prob_errors = (self.scales_ - depth)
        # weighted_variance = (prob * prob_errors).sum(1, keepdims=True)
        # uncert = entropy * F.softplus(self.alpha1) + weighted_variance * F.softplus(self.alpha2)
        
        uncert = entropy * F.softplus(self.alpha1)
        return {'prob':prob, 'depth':depth, 'scales':self.scales, 'entropy':entropy, 'uncert':uncert}

### from AdaBins
class AdaptiveClassificationHead(nn.Module):
    def __init__(self, feat=128, num_class=128, min_val=1e-3, max_val=10):
        super().__init__()
        self.adaptive_bins_layer = mViT(feat, dim_out=num_class)
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, num_class, kernel_size=1, stride=1, padding=0),
            # nn.Softmax(dim=1)
            )
        self.min_val, self.max_val = min_val, max_val
    
    def forward(self, x, temperature):
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(x)
        prob = self.conv_out(range_attention_maps)
        prob = F.softmax(prob * temperature, dim=1)

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        depth = torch.sum(prob * centers, dim=1, keepdim=True)
        entropy = (-prob * torch.log(torch.clamp(prob, 1e-4, 1.0-1e-4))).sum()

        return {'prob':prob, 'depth':depth, 'scales':centers, 'entropy':entropy}


class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        return y, range_attention_maps


class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)
