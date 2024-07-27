import torch
import torch.nn as nn
import math
from lib.ssn.ssn import ssn_iter, sparse_ssn_iter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )

class SSNModel(nn.Module):
    def __init__(self, feature_dim, nspix, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter

        self.scale1 = nn.Sequential(
            conv_bn_relu(5, 64),
            conv_bn_relu(64, 64)
        )
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(64*3+5, feature_dim-5, 3, padding=1),
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        pixel_f = self.feature_extract(x)

        if self.training:
            return ssn_iter(pixel_f, self.nspix, self.n_iter)
        else:
            return sparse_ssn_iter(pixel_f, self.nspix, self.n_iter)


    def feature_extract(self, x, pos_scale=1.0):
        height, width = x.shape[-2:]
        nspix_per_axis = int(math.sqrt(self.nspix))
        pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)
        
        coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
        coords = coords[None].repeat(x.shape[0], 1, 1, 1).float()
        
        x = torch.cat([0.26*x, pos_scale*coords], 1)    
        
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = nn.functional.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = nn.functional.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        Q, H, feat = torch.cat([feat, x], 1)
    
def update_param(data, model, optimizer, compactness, color_scale, pos_scale, device):
    inputs, labels = data

    inputs = inputs.to(device)
    labels = labels.to(device)

    height, width = inputs.shape[-2:]

    nspix_per_axis = int(math.sqrt(model.nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)    

    coords = torch.stack(torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device)), 0)
    coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()

    inputs = torch.cat([color_scale*inputs, pos_scale*coords], 1)

    Q, H, feat = model(inputs)

    recons_loss = reconstruct_loss_with_cross_etnropy(Q, labels)
    compact_loss = reconstruct_loss_with_mse(Q, coords.reshape(*coords.shape[:2], -1), H)

    loss = recons_loss + compactness * compact_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item()}
