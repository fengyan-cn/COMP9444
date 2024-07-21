import torch
from torch import nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


class ResNet50_CBAM(nn.Module):
    def __init__(self):
        super(ResNet50_CBAM, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.cbam_modules = nn.ModuleList([
            CBAMBlock(channel=256),
            CBAMBlock(channel=512),
            CBAMBlock(channel=1024),
            CBAMBlock(channel=2048)
        ])
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256,
        )
    
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c1 = self.cbam_modules[0](c1)
        c2 = self.backbone.layer2(c1)
        c2 = self.cbam_modules[1](c2)
        c3 = self.backbone.layer3(c2)
        c3 = self.cbam_modules[2](c3)
        c4 = self.backbone.layer4(c3)
        c4 = self.cbam_modules[3](c4)
        features = self.fpn({
            '0': c1,
            '1': c2,
            '2': c3,
            '3': c4
        })
        return features

backbone = ResNet50_CBAM()
# Freeze the parameters of the backbone's first few layers
for param in list(backbone.backbone.children())[:5]:
    for p in param.parameters():
        p.requires_grad = False

backbone.out_channels = 256

rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256), (32, 64, 128, 256), (32, 64, 128, 256), (32, 64, 128, 256)),
    aspect_ratios=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)),
)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2,
)

model = FasterRCNN(
    backbone,
    num_classes=7,
    rpn_anchor_generator=rpn_anchor_generator,
    box_roi_pool=roi_pooler,
)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)

# dummy_images = [torch.rand((3, 1024, 1024), device=device) for _ in range(1)]
# dummy_targets = [{
#     "boxes": torch.tensor([[10, 10, 50, 50], [30, 30, 60, 60]], dtype=torch.float32, device=device),
#     "labels": torch.tensor([1, 2], dtype=torch.int64, device=device)
# } for _ in range(1)]

# model.train()

# output = model(dummy_images, dummy_targets)

# print(output)