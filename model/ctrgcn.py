import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y, A


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True,
                 kernel_size=5, dilations=[1, 2], aug=False, layer=1):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        self.aug = aug
        self.layer = layer - 1
        if self.aug:
            self.graph = A
            t_channel = [64, 64, 64, 64, 32, 32, 32, 16, 16, 16]
            self.ca = ChannelAttention(in_planes=out_channels)
            self.sa = SpatialAttention(channel=t_channel[self.layer], graph=self.graph)

    def forward(self, x):
        # y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        x1, A = self.gcn1(x)
        x1 = self.tcn1(x1)
        x2 = self.residual(x)
        if self.aug:
            x1 = self.ca(x1)
            x1 = self.sa(x1, A)
        y = x1 + x2
        y = self.relu(y)

        return y


class SpatialAttention(nn.Module):
    def __init__(self, graph, channel, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.alpha_1 = nn.Parameter(torch.tensor(0.1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, stride=(1, 1), kernel_size=(1, 1), padding=(0, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=1, stride=(1, 1), kernel_size=(1, 5), padding=(0, 0), bias=True),
        )
        self.fc_S = nn.Sequential(
            nn.Conv2d(50, 10, (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 50, (1, 1), bias=True),
        )
        self.fc_A = nn.Sequential(
            nn.Conv2d(2, 9, (1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(9, 1, (1, 1), bias=True),
        )
        self.fc_T = nn.Sequential(
            nn.Conv2d(in_channels=int(channel / 4), out_channels=int(channel / 4), stride=(1, 1), kernel_size=(1, 1),
                      padding=(0, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(channel / 4), out_channels=int(channel / 4), stride=(1, 1), kernel_size=(1, 1),
                      padding=(0, 0), bias=True),
        )
        self.past_A = np.zeros_like([3, 50, 50])
        self.graph = graph  # natural connection

    def forward(self, x, A):
        AA = A.clone().detach()
        n, c, t, v = x.size()  # N C T V 32 64 64 50
        segment = int(t / 4)
        x1 = x[:, :, 0:segment, :]
        x2 = x[:, :, segment:2 * segment, :]
        x3 = x[:, :, 2 * segment:3 * segment, :]
        x4 = x[:, :, 3 * segment:4 * segment, :]
        X = [x1, x2, x3, x4]

        # global attention
        # single_1 = A - self.graph  # 单人动作用
        a_1 = AA[0]  # 50 50
        dif_1 = a_1 - self.past_A[0]
        a_2 = AA[1]
        dif_2 = a_2 - self.past_A[1]
        a_3 = AA[2]
        dif_3 = a_3 - self.past_A[2]
        # AA[:, 0:25, 0:25] = 0  # 双人动作用
        # AA[:, 25:50, 25:50] = 0

        a_1 = torch.cat([a_1.unsqueeze(0), dif_1.unsqueeze(0)], dim=0)
        # a_1 = torch.cat([a_1, AA[0].unsqueeze(0)], dim=0).unsqueeze(0)  # 1 3 50 50
        a_1 = torch.squeeze(self.fc_A(a_1.unsqueeze(0)))

        a_2 = torch.cat([a_2.unsqueeze(0), dif_2.unsqueeze(0)], dim=0)
        # a_2 = torch.cat([a_2, AA[1].unsqueeze(0)], dim=0).unsqueeze(0)  # 1 3 50 50
        a_2 = torch.squeeze(self.fc_A(a_2.unsqueeze(0)))

        a_3 = torch.cat([a_3.unsqueeze(0), dif_3.unsqueeze(0)], dim=0)
        # a_3 = torch.cat([a_3, AA[2].unsqueeze(0)], dim=0).unsqueeze(0)  # 1 3 50 50
        a_3 = torch.squeeze(self.fc_A(a_3.unsqueeze(0)))

        self.pastA = A.clone().detach()  # 保存当前的A
        input_attention_S = []

        for seg in X:
            x_T = seg.permute(0, 2, 1, 3).contiguous()  # N T C V
            avg_out_T = self.fc_T(self.avg_pool(x_T))  # 32 16 1 1
            max_out_T = self.fc_T(self.max_pool(x_T))
            out_T = (avg_out_T + max_out_T) / 2  # 32 16 1 1 代表整体四分之一片段内的时间全局特征
            x_T = x_T * out_T

            x_S = x_T.permute(0, 3, 2, 1).contiguous()  # N V C T
            avg_out_S = self.fc_S(self.avg_pool(x_S))  # 32 50 1 1
            max_out_S = self.fc_S(self.max_pool(x_S))
            out_S = torch.squeeze(avg_out_S + max_out_S) / 2  # 32 25 代表整体四分之一片段内的空间全局特征
            A_1 = torch.matmul(out_S, a_1).unsqueeze(-1) * self.alpha_1
            A_2 = torch.matmul(out_S, a_2).unsqueeze(-1) * self.alpha_1
            A_3 = torch.matmul(out_S, a_3).unsqueeze(-1) * self.alpha_1
            a = torch.cat([A_1, A_2], dim=-1)
            a = torch.cat([a, A_3], dim=-1)
            a = torch.cat([a, torch.squeeze(avg_out_S, dim=-1)], dim=-1)
            a = torch.cat([a, torch.squeeze(max_out_S, dim=-1)], dim=-1)  # 32 25 5
            a = a.unsqueeze(1)  # 32 1 25 5
            out = self.fc_4(a)  # 32 1 25 1
            out = torch.squeeze(out)  # 32 25
            global_out = self.sigmoid(out).view(n, 1, 1, v)  # 32 1 1 25 比例
            x_S = x_S.permute(0, 2, 3, 1).contiguous()  # N C T V

            input_attention_S.append(x_S * global_out)  # 32 64 16 25

        finalout = torch.cat([input_attention_S[0], input_attention_S[1]], dim=2)
        finalout = torch.cat([finalout, input_attention_S[2]], dim=2)
        finalout = torch.cat([finalout, input_attention_S[3]], dim=2)
        return finalout


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.fc_2 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.fc_3 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.fc_4 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.fc_5 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.fc_6 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.fc_7 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.fc_8 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.fc_9 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.fc_10 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, (1, 1), bias=True),
                                   nn.ReLU(),
                                   nn.Conv2d(in_planes // 16, in_planes, (1, 1), bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 两个人单独计算通道注意力
        # Input x = N C T V 16 64 64 50
        people_1_torso = x[:, :, :, [0, 1, 2, 3, 20]]  # torso = N C T V 32 64 64 5
        people_1_left_hand = x[:, :, :, [8, 9, 10, 11, 23, 24]]
        people_1_left_leg = x[:, :, :, [16, 17, 18, 19]]
        people_1_right_hand = x[:, :, :, [4, 5, 6, 7, 21, 22]]
        people_1_right_leg = x[:, :, :, [12, 13, 14, 15]]
        people_2_torso = x[:, :, :, [0+25, 1+25, 2+25, 3+25, 20+25]]  # torso = N C T V 32 64 64 5
        people_2_left_hand = x[:, :, :, [8+25, 9+25, 10+25, 11+25, 23+25, 24+25]]
        people_2_left_leg = x[:, :, :, [16+25, 17+25, 18+25, 19+25]]
        people_2_right_hand = x[:, :, :, [4+25, 5+25, 6+25, 7+25, 21+25, 22+25]]
        people_2_right_leg = x[:, :, :, [12+25, 13+25, 14+25, 15+25]]
        # people_1 = x[:, :, :, 0: 25]  # shape = N C T V 16 64 64 25
        # people_2 = x[:, :, :, 25:50]
        # body = {torso, left_hand, left_leg, right_hand, right_leg}
        avg_out_1 = self.fc_1(self.avg_pool(people_1_torso))  # N C T V 32 64 1 1
        max_out_1 = self.fc_1(self.max_pool(people_1_torso))
        avg_out_2 = self.fc_2(self.avg_pool(people_1_left_hand))  # N C T V 32 64 1 1
        max_out_2 = self.fc_2(self.max_pool(people_1_left_hand))
        avg_out_3 = self.fc_3(self.avg_pool(people_1_left_leg))  # N C T V 32 64 1 1
        max_out_3 = self.fc_3(self.max_pool(people_1_left_leg))
        avg_out_4 = self.fc_4(self.avg_pool(people_1_right_hand))  # N C T V 32 64 1 1
        max_out_4 = self.fc_4(self.max_pool(people_1_right_hand))
        avg_out_5 = self.fc_5(self.avg_pool(people_1_right_leg))  # N C T V 32 64 1 1
        max_out_5 = self.fc_5(self.max_pool(people_1_right_leg))
        avg_out_6 = self.fc_6(self.avg_pool(people_2_torso))  # N C T V 32 64 1 1
        max_out_6 = self.fc_6(self.max_pool(people_2_torso))
        avg_out_7 = self.fc_7(self.avg_pool(people_2_left_hand))  # N C T V 32 64 1 1
        max_out_7 = self.fc_7(self.max_pool(people_2_left_hand))
        avg_out_8 = self.fc_8(self.avg_pool(people_2_left_leg))  # N C T V 32 64 1 1
        max_out_8 = self.fc_8(self.max_pool(people_2_left_leg))
        avg_out_9 = self.fc_9(self.avg_pool(people_2_right_hand))  # N C T V 32 64 1 1
        max_out_9 = self.fc_9(self.max_pool(people_2_right_hand))
        avg_out_10 = self.fc_10(self.avg_pool(people_2_right_leg))  # N C T V 32 64 1 1
        max_out_10 = self.fc_10(self.max_pool(people_2_right_leg))

        out_part_1 = avg_out_1 + max_out_1
        out_part_1 = self.sigmoid(out_part_1)
        people_1_torso = people_1_torso * out_part_1

        out_part_2 = avg_out_2 + max_out_2
        out_part_2 = self.sigmoid(out_part_2)
        people_1_left_hand = people_1_left_hand * out_part_2

        out_part_3 = avg_out_3 + max_out_3
        out_part_3 = self.sigmoid(out_part_3)
        people_1_left_leg = people_1_left_leg * out_part_3

        out_part_4 = avg_out_4 + max_out_4
        out_part_4 = self.sigmoid(out_part_4)
        people_1_right_hand = people_1_right_hand * out_part_4

        out_part_5 = avg_out_5 + max_out_5
        out_part_5 = self.sigmoid(out_part_5)
        people_1_right_leg = people_1_right_leg * out_part_5

        out_part_6 = avg_out_6 + max_out_6
        out_part_6 = self.sigmoid(out_part_6)
        people_2_torso = people_2_torso * out_part_6

        out_part_7 = avg_out_7 + max_out_7
        out_part_7 = self.sigmoid(out_part_7)
        people_2_left_hand = people_2_left_hand * out_part_7

        out_part_8 = avg_out_8 + max_out_8
        out_part_8 = self.sigmoid(out_part_8)
        people_2_left_leg = people_2_left_leg * out_part_8

        out_part_9 = avg_out_9 + max_out_9
        out_part_9 = self.sigmoid(out_part_9)
        people_2_right_hand = people_2_right_hand * out_part_9

        out_part_10 = avg_out_10 + max_out_10
        out_part_10 = self.sigmoid(out_part_10)
        people_2_right_leg = people_2_right_leg * out_part_10

        out = torch.cat([people_1_torso, people_1_left_hand], dim=3)
        out = torch.cat([out, people_1_left_leg], dim=3)
        out = torch.cat([out, people_1_right_hand], dim=3)
        out = torch.cat([out, people_1_right_leg], dim=3)
        out = torch.cat([out, people_2_torso], dim=3)
        out = torch.cat([out, people_2_left_hand], dim=3)
        out = torch.cat([out, people_2_left_leg], dim=3)
        out = torch.cat([out, people_2_right_hand], dim=3)
        out = torch.cat([out, people_2_right_leg], dim=3)

        return out

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc_1 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
#                                   nn.ReLU(),
#                                   nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
#         self.fc_2 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
#                                   nn.ReLU(),
#                                   nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
#         self.fc_3 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
#                                   nn.ReLU(),
#                                   nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
#         self.fc_4 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
#                                   nn.ReLU(),
#                                   nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
#         self.fc_5 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
#                                   nn.ReLU(),
#                                   nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 思路:做细分，每个身体部位单独提取特征来作映射比例，这个比例代表了身体某部分在所有通道中的强弱，然后把比例成回对应的身体部分，再拼接起来
#         # Input x = N C T V 32 64 64 25
#         torso = x[:, :, :, [0, 1, 2, 3, 20]]  # torso = N C T V 32 64 64 5
#         left_hand = x[:, :, :, [8, 9, 10, 11, 23, 24]]
#         left_leg = x[:, :, :, [16, 17, 18, 19]]
#         right_hand = x[:, :, :, [4, 5, 6, 7, 21, 22]]
#         right_leg = x[:, :, :, [12, 13, 14, 15]]
#         # body = {torso, left_hand, left_leg, right_hand, right_leg}
#         avg_out_1 = self.fc_1(self.avg_pool(torso))  # N C T V 32 64 1 1
#         max_out_1 = self.fc_1(self.max_pool(torso))
#         avg_out_2 = self.fc_2(self.avg_pool(left_hand))  # N C T V 32 64 1 1
#         max_out_2 = self.fc_2(self.max_pool(left_hand))
#         avg_out_3 = self.fc_3(self.avg_pool(left_leg))  # N C T V 32 64 1 1
#         max_out_3 = self.fc_3(self.max_pool(left_leg))
#         avg_out_4 = self.fc_4(self.avg_pool(right_hand))  # N C T V 32 64 1 1
#         max_out_4 = self.fc_4(self.max_pool(right_hand))
#         avg_out_5 = self.fc_5(self.avg_pool(right_leg))  # N C T V 32 64 1 1
#         max_out_5 = self.fc_5(self.max_pool(right_leg))
#
#         out_part_1 = avg_out_1 + max_out_1
#         out_part_1 = self.sigmoid(out_part_1)
#         torso = torso * out_part_1
#
#         out_part_2 = avg_out_2 + max_out_2
#         out_part_2 = self.sigmoid(out_part_2)
#         left_hand = left_hand * out_part_2
#
#         out_part_3 = avg_out_3 + max_out_3
#         out_part_3 = self.sigmoid(out_part_3)
#         left_leg = left_leg * out_part_3
#
#         out_part_4 = avg_out_4 + max_out_4
#         out_part_4 = self.sigmoid(out_part_4)
#         right_hand = right_hand * out_part_4
#
#         out_part_5 = avg_out_5 + max_out_5
#         out_part_5 = self.sigmoid(out_part_5)
#         right_leg = right_leg * out_part_5
#
#         out = torch.cat([torso, left_hand], dim=3)
#         out = torch.cat([out, left_leg], dim=3)
#         out = torch.cat([out, right_hand], dim=3)
#         out = torch.cat([out, right_leg], dim=3)
#
#         return out
#

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(base_channel, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, aug=True, layer=3)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive, aug=True, layer=4)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, aug=True, layer=6)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive, aug=True, layer=7)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, aug=True, layer=9)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive, aug=True, layer=10)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
        self.svm = True
        if self.svm:
            self.input_map = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 2, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 2),
                nn.LeakyReLU(0.1),
            )
            self.diff_map1 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map2 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map3 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map4 = nn.Sequential(
                nn.Conv2d(3, base_channel // 8, kernel_size=(1, 1)),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        if self.svm:
            # short-term modeling
            dif1 = x[:, :, 1:, :] - x[:, :, 0:-1, :]  # 16 3 63 50
            dif1 = torch.cat([dif1.new(N * M, C, 1, V).zero_(), dif1], dim=-2)
            dif2 = x[:, :, :-1, :] - x[:, :, 1:, :]
            dif2 = torch.cat([dif2, dif2.new(N * M, C, 1, V).zero_()], dim=-2)
            dif3 = x[:, :, 2:, :] - x[:, :, 0:-2, :]  # 16 3 62 50
            dif3 = torch.cat([dif3.new(N * M, C, 2, V).zero_(), dif3], dim=-2)
            dif4 = x[:, :, :-2, :] - x[:, :, 2:, :]
            dif4 = torch.cat([dif4, dif4.new(N * M, C, 2, V).zero_()], dim=-2)
            x = torch.cat((self.input_map(x), self.diff_map1(dif1), self.diff_map2(dif2),
                           self.diff_map3(dif3), self.diff_map4(dif4)), dim=1)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)