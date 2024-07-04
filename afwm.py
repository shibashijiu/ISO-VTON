import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from options.train_options import TrainOptions

from .correlation import correlation  # the custom cost volume layer
opt = TrainOptions().parse()


def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid(
        [torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
                 for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
                 for grid, size in zip(grid_list, reversed(sizes))]

    return torch.stack(grid_list, dim=-1)


def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))

"""
def TVLoss_v2(x, mask):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    h, w = mask.size(2), mask.size(3)

    tv_h = tv_h * mask[:, :, :h-1, :]
    tv_w = tv_w * mask[:, :, :, :w-1]

    if torch.sum(mask) > 0:
        return (torch.sum(torch.abs(tv_h)) + torch.sum(torch.abs(tv_w))) / torch.sum(mask)
    else:
        return torch.sum(torch.abs(tv_h)) + torch.sum(torch.abs(tv_w))


def SquareTVLoss(flow):
    flow_x, flow_y = torch.split(flow, 1, dim=1)

    flow_x_diff_left = flow_x[:, :, :, 1:] - flow_x[:, :, :, :-1]
    flow_x_diff_right = flow_x[:, :, :, :-1] - flow_x[:, :, :, 1:]
    flow_x_diff_left = flow_x_diff_left[...,1:-1,:-1]
    flow_x_diff_right = flow_x_diff_right[...,1:-1,1:]

    flow_y_diff_top = flow_y[:, :, 1:, :] - flow_y[:, :, :-1, :]
    flow_y_diff_bottom = flow_y[:, :, :-1, :] - flow_y[:, :, 1:, :]
    flow_y_diff_top = flow_y_diff_top[...,:-1,1:-1]
    flow_y_diff_bottom = flow_y_diff_bottom[...,1:,1:-1]

    left_top_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_top))
    left_bottom_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_bottom))
    right_top_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_top))
    right_bottom_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_bottom))

    return torch.mean(left_top_diff+left_bottom_diff+right_top_diff+right_bottom_diff)

def SquareTVLoss_v2(flow, interval_list=[1,5]):
    flow_x, flow_y = torch.split(flow, 1, dim=1)

    tvloss = 0
    for interval in interval_list:
        flow_x_diff_left = flow_x[:, :, :, interval:] - flow_x[:, :, :, :-interval]
        flow_x_diff_right = flow_x[:, :, :, :-interval] - flow_x[:, :, :, interval:]
        flow_x_diff_left = flow_x_diff_left[...,interval:-interval,:-interval]
        flow_x_diff_right = flow_x_diff_right[...,interval:-interval,interval:]

        flow_y_diff_top = flow_y[:, :, interval:, :] - flow_y[:, :, :-interval, :]
        flow_y_diff_bottom = flow_y[:, :, :-interval, :] - flow_y[:, :, interval:, :]
        flow_y_diff_top = flow_y_diff_top[...,:-interval,interval:-interval]
        flow_y_diff_bottom = flow_y_diff_bottom[...,interval:,interval:-interval]

        left_top_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_top))
        left_bottom_diff = torch.abs(torch.abs(flow_x_diff_left) - torch.abs(flow_y_diff_bottom))
        right_top_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_top))
        right_bottom_diff = torch.abs(torch.abs(flow_x_diff_right) - torch.abs(flow_y_diff_bottom))

        tvloss += torch.mean(left_top_diff+left_bottom_diff+right_top_diff+right_bottom_diff)

    return tvloss
"""







#权重的等化学习率，定义stylegan模块作为style特征提取块
# backbone 这是一个用于权重标准化的类，通常用于StyleGAN网络中。其目的是调整卷积层或全连接层的权重，使得学习率在所有层中保持一致。
class EqualLR:
    def __init__(self, name): # 初始化，保存权重的名称。
        self.name = name

    def compute_weight(self, module): # 根据权重的fan-in（输入神经元数量）计算标准化后的权重。
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name): # 将EqualLR应用于指定的模块，修改其权重，并注册前向传播钩子。
        fn = EqualLR(name) 

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input): # 在模块的前向传播过程中调用，用于调整权重。
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

# 使用上述调的类
def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class ModulatedConv2d(nn.Module):
    def __init__(self, fin, fout, kernel_size, padding_type='zero', upsample=False, downsample=False, latent_dim=512, normalize_mlp=False):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = fin
        self.out_channels = fout
        self.kernel_size = kernel_size
        padding_size = kernel_size // 2

        if kernel_size == 1:
            self.demudulate = False
        else:
            self.demudulate = True

        self.weight = nn.Parameter(torch.Tensor(fout, fin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1, fout, 1, 1))
        #self.conv = F.conv2d

        if normalize_mlp:
            self.mlp_class_std = nn.Sequential(EqualLinear(latent_dim, fin), PixelNorm())
        else:
            self.mlp_class_std = EqualLinear(latent_dim, fin)

        #self.blur = Blur(fout)

        if padding_type == 'reflect':
            self.padding = nn.ReflectionPad2d(padding_size)
        else:
            self.padding = nn.ZeroPad2d(padding_size)


        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, input, latent):
        fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        weight = self.weight * sqrt(2 / fan_in)
        weight = weight.view(1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        s = self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
        weight = s * weight
        if self.demudulate:
            d = torch.rsqrt((weight ** 2).sum(4).sum(3).sum(2) + 1e-5).view(-1, self.out_channels, 1, 1, 1)
            weight = (d * weight).view(-1, self.in_channels, self.kernel_size, self.kernel_size)
        else:
            weight = weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        

        batch,_,height,width = input.shape
        #input = input.view(1,-1,h,w)
        #input = self.padding(input)
        #out = self.conv(input, weight, groups=b).view(b, self.out_channels, h, w) + self.bias

        

        input = input.view(1,-1,height,width)
        input = self.padding(input)
        out = F.conv2d(input, weight, groups=batch).view(batch, self.out_channels, height, width) + self.bias

        return out


class StyledConvBlock(nn.Module):
    def __init__(self, fin, fout, latent_dim=256, padding='zero',
                 actvn='lrelu', normalize_affine_output=False, modulated_conv=False):
        super(StyledConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        
        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2,True)


        if self.modulated_conv:
            self.conv0 = conv2d(fin, fout, kernel_size=3, padding_type=padding, upsample=False,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, fout, kernel_size=3)
  
            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(fout, fout, kernel_size=3, padding_type=padding, downsample=False,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(fout, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input,latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out,latent)
        else:
            out = self.conv1(out)

        out = self.actvn1(out) * self.actvn_gain

        return out


class Styled_F_ConvBlock(nn.Module):
    def __init__(self, fin, fout, latent_dim=256, padding='zero',
                 actvn='lrelu', normalize_affine_output=False, modulated_conv=False):
        super(Styled_F_ConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        
        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2,True)


        if self.modulated_conv:
            self.conv0 = conv2d(fin, 128, kernel_size=3, padding_type=padding, upsample=False,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, 128, kernel_size=3)
  
            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(128, fout, kernel_size=3, padding_type=padding, downsample=False,
                                latent_dim=latent_dim, normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(128, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        #self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input,latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out,latent)
        else:
            out = self.conv1(out)

        #out = self.actvn1(out) * self.actvn_gain

        return out









# backbone

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x) + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        return self.block(x)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i-1], out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features


class RefinePyramid(nn.Module):
    def __init__(self, chns=[64, 128, 256, 256, 256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive
        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(
                fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x

        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + \
                    F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))


class AFlowNet_Vitonhd_lrarms(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super(AFlowNet_Vitonhd_lrarms, self).__init__()

        #style中的参数
        padding_type='zero'
        actvn = 'lrelu'
        normalize_mlp = False
        modulated_conv = True
        #style中的参数

        #加的两个列表
        self.netStyle = []

        self.netLeftF = []
        self.netTorsoF = []
        self.netRightF = []
        #加的两个列表
        
        #self.netLeftMain = []
        #self.netTorsoMain = []
        #self.netRightMain = []

        self.netLeftRefine = []
        self.netTorsoRefine = []
        self.netRightRefine = []

        self.netAttentionRefine = []
        self.netPartFusion = []
        self.netSeg = []

        for i in range(num_pyramid):
            #定义style风格卷积块，输入256，输出49
            style_block = StyledConvBlock(256, 49, latent_dim=256,
                                         padding=padding_type, actvn=actvn,
                                         normalize_affine_output=normalize_mlp,
                                         modulated_conv=modulated_conv)
            #定义style风格卷积块，输入256，输出49
    
            #定义style卷积块，输入49，输出2
            style_F_block_Left = Styled_F_ConvBlock(49, 2, latent_dim=256,
                                              padding=padding_type, actvn=actvn,
                                              normalize_affine_output=normalize_mlp,
                                              modulated_conv=modulated_conv)   
            #定义style卷积块，输入49，输出2

            style_F_block_Torso = Styled_F_ConvBlock(49, 2, latent_dim=256,
                                              padding=padding_type, actvn=actvn,
                                              normalize_affine_output=normalize_mlp,
                                              modulated_conv=modulated_conv)   
            #定义style卷积块，输入49，输出2

            style_F_block_Right = Styled_F_ConvBlock(49, 2, latent_dim=256,
                                              padding=padding_type, actvn=actvn,
                                              normalize_affine_output=normalize_mlp,
                                              modulated_conv=modulated_conv)   
            #定义style卷积块，输入49，输出2           
            
            #main流的输出和refine流的输出都是2通道，输入main是49通道，refine是256通道


            netRefine_left_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRefine_torso_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRefine_right_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            )
            #AttentionRefine_layer输入为1024通道，输出为3通道
            netAttentionRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=4 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=3,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
            #AttentionRefine_layer输入为1024通道，输出为3通道
            #Seg_layer输入为512通道，输出为7通道
            netSeg_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=fpn_dim*2, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=7,  #这里是7
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
            #Seg_layer输入为512通道，输出为7通道
            #Fusion_layer输入为768通道，经过ResBlock输出也为256通道
            partFusion_layer = torch.nn.Sequential(
                nn.Conv2d(fpn_dim*3, fpn_dim, kernel_size=1),
                ResBlock(fpn_dim)
            )
            #Fusion_layer输入为768通道，经过ResBlock输出也为256通道
            self.netStyle.append(style_block)

            self.netLeftF.append(style_F_block_Left)
            self.netTorsoF.append(style_F_block_Torso)
            self.netRightF.append(style_F_block_Right)


            self.netLeftRefine.append(netRefine_left_layer)
            self.netTorsoRefine.append(netRefine_torso_layer)
            self.netRightRefine.append(netRefine_right_layer)

            self.netAttentionRefine.append(netAttentionRefine_layer)
            self.netPartFusion.append(partFusion_layer)
            self.netSeg.append(netSeg_layer)

            #添加的两个风格块


        self.netLeftRefine = nn.ModuleList(self.netLeftRefine)
        self.netTorsoRefine = nn.ModuleList(self.netTorsoRefine)
        self.netRightRefine = nn.ModuleList(self.netRightRefine)

        self.netAttentionRefine = nn.ModuleList(self.netAttentionRefine)
        self.netPartFusion = nn.ModuleList(self.netPartFusion)
        self.netSeg = nn.ModuleList(self.netSeg)
        
        self.softmax = torch.nn.Softmax(dim=1) #该函数在dim=1维度，也就是通道维度上进行概率分布，但是整体的张量大小不会改变
          
        #每次采集的信息添加到列表中
        self.netStyle = nn.ModuleList(self.netStyle)

        self.netLeftF = nn.ModuleList(self.netLeftF)
        self.netTorsoF = nn.ModuleList(self.netTorsoF)
        self.netRightF = nn.ModuleList(self.netRightF)
        #每次采集的信息添加到列表中
        #定义的两个模块，一个作用于衣服，一个作用于人体表征
        self.cond_style = torch.nn.Sequential(torch.nn.Conv2d(256, 128, kernel_size=(16,12), stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
     
        self.image_style = torch.nn.Sequential(torch.nn.Conv2d(256, 128, kernel_size=(16,12), stride=1, padding=0), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
        #定义的两个模块，一个作用于衣服，一个作用于人体表征

         

    def forward(self, x, x_edge, x_full, x_edge_full, x_warps, x_conds, preserve_mask, warp_feature=True):
        last_flow = None
        last_flow_all = []
        delta_list = []
        x_all = []
        x_edge_all = []

        x_full_all = []
        x_edge_full_all = []
        attention_all = []
        seg_list = []

        delta_x_all = []
        delta_y_all = []
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array = torch.cuda.FloatTensor(weight_array).permute(3, 2, 0, 1)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)


  
        #新加的部分
        B = x_conds[len(x_warps)-1].shape[0]  #确定列表中最后一个多维数组的batch size，也就是特征p4的bs
        cond_style = self.cond_style(x_conds[len(x_warps) - 1]).view(B,-1)  #cond_style为单独变量
        image_style = self.image_style(x_warps[len(x_warps) - 1]).view(B,-1)  #image_style为单独变量
        style = torch.cat([cond_style, image_style], 1)  #style也为单独变量
        style_concate = torch.cat([style,style,style],0)
        #上述这部分GP-VTON没有
        # fp全连接层，将最后一个特征p4（多维数组）送入到self.cond_style函数层中进行处理torch.nn.Conv2d(256, 128),通道数减半,输出特征为(8,128,57,59)
        #两个view后为(8, 430848)，故style=(8, 861696)。
        
        for i in range(len(x_warps)):
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            #特征图复制（Feature Map Replication）：如果是沿着特征图维度复制，则最终的特征维度将会是（8 * 3, 256, 64, 64），即特征图数量增加到原来的三倍。
            x_cond_concate = torch.cat([x_cond,x_cond,x_cond],0) 
            x_warp_concate = torch.cat([x_warp,x_warp,x_warp],0)
            
            if last_flow is not None and warp_feature:
                x_warp_after = F.grid_sample(x_warp_concate, last_flow.detach().permute(0, 2, 3, 1),  #修改了x_warp_after = x_warp_concate
                                             mode='bilinear', padding_mode='border')
            else:
                x_warp_after = x_warp_concate  #修改了x_warp_after = x_warp_concate
            
            #新加的 x_warp_after(24, 256, 64, 64) style_concate为(24, 861696) 求相关性得到(24, 49, 64, 64)
            stylemap = self.netStyle[i](x_warp_after, style_concate)  #单独的,相当于求tenCorrelation
            bz = x_cond.size(0)
            left_stylemap = stylemap[0:bz]
            torso_stylemap = stylemap[bz:2*bz]
            right_stylemap = stylemap[2*bz:]
            
            left_flow = self.netLeftF[i](left_stylemap, style)  #单独的
            torso_flow = self.netTorsoF[i](torso_stylemap, style)
            right_flow = self.netRightF[i](right_stylemap, style)
            
            flow = torch.cat([left_flow, torso_flow, right_flow],0)
            delta_list.append(flow)  #添加在delta_list中
            flow = apply_offset(flow)  #单独的风格外观流
            #f(ci)
            #新加的
            """ #如果两个大小均为 (24, 256, 64, 64)，经过tenCorrelation操作我们可以得到的大小将会是(24, 49, 64, 64)
            tenCorrelation = F.leaky_relu(input=correlation.FunctionCorrelation(
                tenFirst=x_warp_after, tenSecond=x_cond_concate, intStride=1), negative_slope=0.1, inplace=False)
            

            bz = x_cond.size(0)

            left_tenCorrelation = tenCorrelation[0:bz]
            torso_tenCorrelation = tenCorrelation[bz:2*bz]
            right_tenCorrelation = tenCorrelation[2*bz:]

            left_flow = self.netLeftMain[i](left_tenCorrelation)
            torso_flow = self.netTorsoMain[i](torso_tenCorrelation)
            right_flow = self.netRightMain[i](right_tenCorrelation)
            
            flow = torch.cat([left_flow,torso_flow,right_flow],0)
            
            delta_list.append(flow)
            flow = apply_offset(flow)
            """
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')
            else:
                flow = flow.permute(0, 3, 1, 2)

            last_flow = flow
            #x_warp为g4，flow为f(ci)
            x_warp_concate = F.grid_sample(x_warp_concate, flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')

            left_concat = torch.cat([x_warp_concate[0:bz], x_cond_concate[0:bz]], 1)
            torso_concat = torch.cat([x_warp_concate[bz:2*bz], x_cond_concate[bz:2*bz]],1)
            right_concat = torch.cat([x_warp_concate[2*bz:], x_cond_concate[2*bz:]],1)

            x_attention = torch.cat([x_warp_concate[0:bz],x_warp_concate[bz:2*bz],x_warp_concate[2*bz:],x_cond],1)
            fused_attention = self.netAttentionRefine[i](x_attention)
            fused_attention = self.softmax(fused_attention)
            
            left_flow = self.netLeftRefine[i](left_concat)
            torso_flow = self.netTorsoRefine[i](torso_concat)
            right_flow = self.netRightRefine[i](right_concat)   

            flow = torch.cat([left_flow,torso_flow,right_flow],0)  
            delta_list.append(flow)
            flow = apply_offset(flow)   #偏移操作获得#fri
            flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')  #Addition
            #fi

            fused_flow = flow[0:bz] * fused_attention[:,0:1,...] + \
                         flow[bz:2*bz] * fused_attention[:,1:2,...] + \
                         flow[2*bz:] * fused_attention[:,2:3,...]
            last_fused_flow = F.interpolate(fused_flow, scale_factor=2, mode='bilinear')

            fused_attention = F.interpolate(fused_attention, scale_factor=2, mode='bilinear')
            attention_all.append(fused_attention)

            cur_x_full = F.interpolate(x_full, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_full_warp = F.grid_sample(cur_x_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_full_all.append(cur_x_full_warp)
            cur_x_edge_full = F.interpolate(x_edge_full, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_edge_full_warp = F.grid_sample(cur_x_edge_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_full_all.append(cur_x_edge_full_warp)

            last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)

            cur_x = F.interpolate(x, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_warp = F.grid_sample(cur_x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_all.append(cur_x_warp)
            cur_x_edge = F.interpolate(x_edge, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_warp_edge = F.grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_all.append(cur_x_warp_edge)

            flow_x, flow_y = torch.split(last_flow, 1, dim=1)
            delta_x = F.conv2d(flow_x, self.weight)
            delta_y = F.conv2d(flow_y, self.weight)
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

            # predict seg
            cur_preserve_mask = F.interpolate(preserve_mask, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_warp = torch.cat([x_warp,x_warp,x_warp],0)
            x_warp = F.interpolate(x_warp, scale_factor=2, mode='bilinear')
            x_cond = F.interpolate(x_cond, scale_factor=2, mode='bilinear')

            x_warp = F.grid_sample(x_warp, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')
            x_warp_left = x_warp[0:bz]
            x_warp_torso = x_warp[bz:2*bz]
            x_warp_right = x_warp[2*bz:]

            x_edge_left = cur_x_warp_edge[0:bz]
            x_edge_torso = cur_x_warp_edge[bz:2*bz]
            x_edge_right = cur_x_warp_edge[2*bz:]

            x_warp_left = x_warp_left * x_edge_left * (1-cur_preserve_mask)
            x_warp_torso = x_warp_torso * x_edge_torso * (1-cur_preserve_mask)
            x_warp_right = x_warp_right * x_edge_right * (1-cur_preserve_mask)

            x_warp = torch.cat([x_warp_left,x_warp_torso,x_warp_right],1)
            x_warp = self.netPartFusion[i](x_warp)

            concate = torch.cat([x_warp,x_cond],1)
            seg = self.netSeg[i](concate)
            seg_list.append(seg)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, x_full_all, \
                x_edge_full_all, attention_all, seg_list


class AFWM_Vitonhd_lrarms(nn.Module):
    def __init__(self, opt, input_nc, clothes_input_nc=3):
        super(AFWM_Vitonhd_lrarms, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        # num_filters = [64,128,256,512,512]
        fpn_dim = 256
        self.image_features = FeatureEncoder(clothes_input_nc+1, num_filters)
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        self.cond_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        
        self.aflow_net = AFlowNet_Vitonhd_lrarms(len(num_filters))
        self.old_lr = opt.lr
        self.old_lr_warp = opt.lr*0.2

    def forward(self, cond_input, image_input, image_edge, image_label_input, image_input_left, image_input_torso, \
                image_input_right, image_edge_left, image_edge_torso, image_edge_right, preserve_mask):
        image_input_concat = torch.cat([image_input, image_label_input],1)

        image_pyramids = self.image_FPN(self.image_features(image_input_concat))
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))  # maybe use nn.Sequential

        image_concat = torch.cat([image_input_left,image_input_torso,image_input_right],0)
        image_edge_concat = torch.cat([image_edge_left, image_edge_torso, image_edge_right],0)

        last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
            x_full_all, x_edge_full_all, attention_all, seg_list = self.aflow_net(image_concat, \
            image_edge_concat, image_input, image_edge, image_pyramids, cond_pyramids, \
            preserve_mask)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
                x_full_all, x_edge_full_all, attention_all, seg_list

    def update_learning_rate(self, optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate_warp(self, optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr


### 混合所有衣服类别，flow预测网络引入spade区别不同类别衣服，衣服特征提取也使用SPADE，增加不同类别特征间的判别性
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, stride=1, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class ResBlock_SPADE(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock_SPADE, self).__init__()
        self.norm_0 = SPADE(in_channels,1)
        self.norm_1 = SPADE(in_channels,1)

        self.actvn_0 = nn.LeakyReLU(inplace=False, negative_slope=0.1)
        self.actvn_1 = nn.LeakyReLU(inplace=False, negative_slope=0.1)

        self.conv_0 = nn.Conv2d(in_channels,in_channels,kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels,in_channels,kernel_size=3, padding=1)

    def forward(self, x, label_map):
        dx = self.conv_0(self.actvn_0(self.norm_0(x, label_map)))
        dx = self.conv_1(self.actvn_1(self.norm_1(dx, label_map)))

        return dx + x


class FeatureEncoder_SPADE(nn.Module):
    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]):
        super(FeatureEncoder_SPADE, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.ModuleList([DownSample(in_channels, out_chns),
                                        ResBlock_SPADE(out_chns),
                                        ResBlock_SPADE(out_chns)])
            else:
                encoder = nn.ModuleList([DownSample(chns[i-1], out_chns),
                                        ResBlock_SPADE(out_chns),
                                        ResBlock_SPADE(out_chns)])

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x, label_map):
        encoder_features = []
        for encoder in self.encoders:
            for ii, encoder_submodule in enumerate(encoder):
                if ii == 0:
                    x = encoder_submodule(x)
                else:
                    x = encoder_submodule(x, label_map)
            encoder_features.append(x)
        return encoder_features



class AFlowNet_Dresscode_lrarms(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super(AFlowNet_Dresscode_lrarms, self).__init__()
        self.netLeftMain = []
        self.netTorsoMain = []
        self.netRightMain = []

        self.netLeftRefine = []
        self.netTorsoRefine = []
        self.netRightRefine = []

        self.netAttentionRefine = []
        self.netPartFusion = []
        self.netSeg = []

        for i in range(num_pyramid):
            netLeftMain_layer = nn.ModuleList([
                torch.nn.Conv2d(in_channels=49, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            ])
            netTorsoMain_layer = nn.ModuleList([
                torch.nn.Conv2d(in_channels=49, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            ])
            netRightMain_layer = nn.ModuleList([
                torch.nn.Conv2d(in_channels=49, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            ])

            netRefine_left_layer = nn.ModuleList([
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            ])
            netRefine_torso_layer = nn.ModuleList([
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            ])
            netRefine_right_layer = nn.ModuleList([
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2,
                                kernel_size=3, stride=1, padding=1)
            ])

            netAttentionRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=4 * fpn_dim, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=3,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

            netSeg_layer = nn.ModuleList([
                torch.nn.Conv2d(in_channels=fpn_dim*2, out_channels=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=10,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            ])

            partFusion_layer = torch.nn.Sequential(
                nn.Conv2d(fpn_dim*3, fpn_dim, kernel_size=1),
                ResBlock(fpn_dim)
            )

            self.netLeftMain.append(netLeftMain_layer)
            self.netTorsoMain.append(netTorsoMain_layer)
            self.netRightMain.append(netRightMain_layer)

            self.netLeftRefine.append(netRefine_left_layer)
            self.netTorsoRefine.append(netRefine_torso_layer)
            self.netRightRefine.append(netRefine_right_layer)

            self.netAttentionRefine.append(netAttentionRefine_layer)
            self.netPartFusion.append(partFusion_layer)
            self.netSeg.append(netSeg_layer)

        self.netLeftMain = nn.ModuleList(self.netLeftMain)
        self.netTorsoMain = nn.ModuleList(self.netTorsoMain)
        self.netRightMain = nn.ModuleList(self.netRightMain)

        self.netLeftRefine = nn.ModuleList(self.netLeftRefine)
        self.netTorsoRefine = nn.ModuleList(self.netTorsoRefine)
        self.netRightRefine = nn.ModuleList(self.netRightRefine)

        self.netAttentionRefine = nn.ModuleList(self.netAttentionRefine)
        self.netPartFusion = nn.ModuleList(self.netPartFusion)
        self.netSeg = nn.ModuleList(self.netSeg)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x, x_edge, x_full, x_edge_full, x_warps, x_conds, preserve_mask, cloth_label_map, warp_feature=True):
        last_flow = None
        last_flow_all = []
        delta_list = []
        x_all = []
        x_edge_all = []
        x_full_all = []
        x_edge_full_all = []
        attention_all = []
        seg_list = []
        delta_x_all = []
        delta_y_all = []
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array = torch.cuda.FloatTensor(weight_array).permute(3, 2, 0, 1)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        for i in range(len(x_warps)):
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_cond_concate = torch.cat([x_cond,x_cond,x_cond],0)
            x_warp_concate = torch.cat([x_warp,x_warp,x_warp],0)

            if last_flow is not None and warp_feature:
                x_warp_after = F.grid_sample(x_warp_concate, last_flow.detach().permute(0, 2, 3, 1),
                                             mode='bilinear', padding_mode='border')
            else:
                x_warp_after = x_warp_concate

            tenCorrelation = F.leaky_relu(input=correlation.FunctionCorrelation(
                tenFirst=x_warp_after, tenSecond=x_cond_concate, intStride=1), negative_slope=0.1, inplace=False)
            
            bz = x_cond.size(0)

            left_flow = tenCorrelation[0:bz]
            torso_flow = tenCorrelation[bz:2*bz]
            right_flow = tenCorrelation[2*bz:]

            for ii, sub_flow_module in enumerate(self.netLeftMain[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    left_flow = sub_flow_module(left_flow, cloth_label_map)
                else:
                    left_flow = sub_flow_module(left_flow)

            for ii, sub_flow_module in enumerate(self.netTorsoMain[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    torso_flow = sub_flow_module(torso_flow, cloth_label_map)
                else:
                    torso_flow = sub_flow_module(torso_flow)

            for ii, sub_flow_module in enumerate(self.netRightMain[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    right_flow = sub_flow_module(right_flow, cloth_label_map)
                else:
                    right_flow = sub_flow_module(right_flow)

            flow = torch.cat([left_flow,torso_flow,right_flow],0)

            delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')
            else:
                flow = flow.permute(0, 3, 1, 2)

            last_flow = flow
            x_warp_concate = F.grid_sample(x_warp_concate, flow.permute(
                0, 2, 3, 1), mode='bilinear', padding_mode='border')

            left_concat = torch.cat([x_warp_concate[0:bz], x_cond_concate[0:bz]], 1)
            torso_concat = torch.cat([x_warp_concate[bz:2*bz], x_cond_concate[bz:2*bz]],1)
            right_concat = torch.cat([x_warp_concate[2*bz:], x_cond_concate[2*bz:]],1)

            x_attention = torch.cat([x_warp_concate[0:bz],x_warp_concate[bz:2*bz],x_warp_concate[2*bz:],x_cond],1)
            fused_attention = self.netAttentionRefine[i](x_attention)
            fused_attention = self.softmax(fused_attention)

            left_flow = left_concat
            torso_flow = torso_concat
            right_flow = right_concat

            for ii, sub_flow_module in enumerate(self.netLeftRefine[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    left_flow = sub_flow_module(left_flow, cloth_label_map)
                else:
                    left_flow = sub_flow_module(left_flow)

            for ii, sub_flow_module in enumerate(self.netTorsoRefine[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    torso_flow = sub_flow_module(torso_flow, cloth_label_map)
                else:
                    torso_flow = sub_flow_module(torso_flow)

            for ii, sub_flow_module in enumerate(self.netRightRefine[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    right_flow = sub_flow_module(right_flow, cloth_label_map)
                else:
                    right_flow = sub_flow_module(right_flow)     

            flow = torch.cat([left_flow,torso_flow,right_flow],0)

            delta_list.append(flow)
            flow = apply_offset(flow)
            flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')

            fused_flow = flow[0:bz] * fused_attention[:,0:1,...] + \
                         flow[bz:2*bz] * fused_attention[:,1:2,...] + \
                         flow[2*bz:] * fused_attention[:,2:3,...]
            last_fused_flow = F.interpolate(fused_flow, scale_factor=2, mode='bilinear')

            fused_attention = F.interpolate(fused_attention, scale_factor=2, mode='bilinear')
            attention_all.append(fused_attention)

            cur_x_full = F.interpolate(x_full, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_full_warp = F.grid_sample(cur_x_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_full_all.append(cur_x_full_warp)
            cur_x_edge_full = F.interpolate(x_edge_full, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_edge_full_warp = F.grid_sample(cur_x_edge_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_full_all.append(cur_x_edge_full_warp)

            last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)

            cur_x = F.interpolate(x, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_warp = F.grid_sample(cur_x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_all.append(cur_x_warp)
            cur_x_edge = F.interpolate(x_edge, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_warp_edge = F.grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_all.append(cur_x_warp_edge)

            flow_x, flow_y = torch.split(last_flow, 1, dim=1)
            delta_x = F.conv2d(flow_x, self.weight)
            delta_y = F.conv2d(flow_y, self.weight)
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

            # predict seg
            cur_preserve_mask = F.interpolate(preserve_mask, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_warp = torch.cat([x_warp,x_warp,x_warp],0)
            x_warp = F.interpolate(x_warp, scale_factor=2, mode='bilinear')
            x_cond = F.interpolate(x_cond, scale_factor=2, mode='bilinear')

            x_warp = F.grid_sample(x_warp, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')
            x_warp_left = x_warp[0:bz]
            x_warp_torso = x_warp[bz:2*bz]
            x_warp_right = x_warp[2*bz:]

            x_edge_left = cur_x_warp_edge[0:bz]
            x_edge_torso = cur_x_warp_edge[bz:2*bz]
            x_edge_right = cur_x_warp_edge[2*bz:]

            x_warp_left = x_warp_left * x_edge_left * (1-cur_preserve_mask)
            x_warp_torso = x_warp_torso * x_edge_torso * (1-cur_preserve_mask)
            x_warp_right = x_warp_right * x_edge_right * (1-cur_preserve_mask)

            x_warp = torch.cat([x_warp_left,x_warp_torso,x_warp_right],1)
            x_warp = self.netPartFusion[i](x_warp)

            seg = torch.cat([x_warp, x_cond],1)
            for ii, sub_flow_module in enumerate(self.netSeg[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    seg = sub_flow_module(seg, cloth_label_map)
                else:
                    seg = sub_flow_module(seg)
            seg_list.append(seg)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, x_full_all, \
                x_edge_full_all, attention_all, seg_list


class AFWM_Dressescode_lrarms(nn.Module):
    def __init__(self, opt, input_nc, clothes_input_nc=3):
        super(AFWM_Dressescode_lrarms, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        # num_filters = [64,128,256,512,512]
        fpn_dim = 256
        self.image_features = FeatureEncoder_SPADE(clothes_input_nc+1, num_filters)
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        self.cond_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        
        self.aflow_net = AFlowNet_Dresscode_lrarms(len(num_filters))
        self.old_lr = opt.lr
        self.old_lr_warp = opt.lr*0.2

    def forward(self, cond_input, image_input, image_edge, image_label_input, image_input_left, image_input_torso, \
                image_input_right, image_edge_left, image_edge_torso, image_edge_right, preserve_mask, cloth_label_map):
        image_input_concat = torch.cat([image_input, image_label_input],1)

        image_pyramids = self.image_FPN(self.image_features(image_input_concat, cloth_label_map))
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))  # maybe use nn.Sequential

        image_concat = torch.cat([image_input_left,image_input_torso,image_input_right],0)
        image_edge_concat = torch.cat([image_edge_left, image_edge_torso, image_edge_right],0)

        last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
            x_full_all, x_edge_full_all, attention_all, seg_list = self.aflow_net(image_concat, \
            image_edge_concat, image_input, image_edge, image_pyramids, cond_pyramids, \
            preserve_mask, cloth_label_map)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
                x_full_all, x_edge_full_all, attention_all, seg_list

    def update_learning_rate(self, optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate_warp(self, optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr
