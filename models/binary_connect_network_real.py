import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
__all__ = ['VGG_like', 'vgg_like']


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class RealConv2d_binact(nn.Module):  # change the name of RealConv2d
    def __init__(self, input_channels, output_channels, M=5, N=1,
                 kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
                 Linear=False):
        super(RealConv2d_binact, self).__init__()
        self.layer_type = 'RealConv2d'
        self.M = M
        self.N = N
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)

        self.beta = nn.ParameterList([])
        for i in range(self.N):
            self.beta.append(nn.Parameter(torch.Tensor([1])))
        self.shift = nn.ParameterList([])
        a = -1
        if self.N==5:
            a=-3
        for i in range(self.N):
            self.shift.append(nn.Parameter(torch.Tensor([a])))
            a = a + 1

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        dpout = BinActive()(x + self.shift[0]) * self.beta[0]
        for i in range(1, self.N):
            dpout = dpout + BinActive()(x + self.shift[i]) * self.beta[i]
        if self.dropout_ratio != 0:
            dpout = self.dropout(dpout)
        if not self.Linear:
            convout = self.conv(dpout)
        else:
            convout = self.linear(dpout)
        convout = self.relu(convout)
        return convout


class Vgg_like_real_binact(nn.Module):
    def __init__(self, num_classes=10, M=5, N=1):
        super(Vgg_like_real_binact, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.N = N
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bconv2 = RealConv2d_binact(128, 128, M=self.M, N=self.N, kernel_size=3, stride=1, padding=1)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bconv3 = RealConv2d_binact(128, 256, M=self.M, N=self.N, kernel_size=3, stride=1, padding=1)
        self.bconv4 = RealConv2d_binact(256, 256, M=self.M, N=self.N, kernel_size=3, stride=1, padding=1)
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bconv5 = RealConv2d_binact(256, 512, M=self.M, N=self.N, kernel_size=3, stride=1, padding=1)
        self.bconv6 = RealConv2d_binact(512, 512, M=self.M, N=self.N, kernel_size=3, stride=1, padding=1)
        self.mpool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc7 = RealConv2d_binact(512 * 4 * 4, 1024, M=self.M, N=self.N, Linear=True)
        self.fc8 = RealConv2d_binact(1024, 1024, M=self.M, N=self.N, dropout=0.5, Linear=True)
        self.bn8 = nn.BatchNorm1d(1024, eps=1e-3, momentum=0.1, affine=True)
        self.dp8 = nn.Dropout()
        self.fc9 = nn.Linear(1024, num_classes)

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            10: {'lr': 5e-3},
            15: {'lr': 1e-3, 'weight_decay': 0},
            20: {'lr': 5e-4},
            25: {'lr': 1e-4}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def forward(self, input):
        conv1out = self.conv1(input)
        bn1out = self.bn1(conv1out)
        relu1out = self.relu1(bn1out)
        bconv2out = self.bconv2(relu1out)
        mpool2out = self.mpool2(bconv2out)
        bconv3out = self.bconv3(mpool2out)
        bconv4out = self.bconv4(bconv3out)
        mpool4out = self.mpool4(bconv4out)
        bconv5out = self.bconv5(mpool4out)
        bconv6out = self.bconv6(bconv5out)
        mpool6out = self.mpool6(bconv6out)
        mpool6out = mpool6out.view(mpool6out.size(0), 512 * 4 * 4)
        fc7out = self.fc7(mpool6out)
        fc8out = self.fc8(fc7out)
        bn8out = self.bn8(fc8out)
        dp8out = self.dp8(bn8out)
        fc9out = self.fc9(dp8out)
        return fc9out#, bn1out, bconv2out, bconv3out, bconv4out, bconv5out, bconv6out, fc7out, fc8out


def vgg_like_real_binact(**kwargs):
    """model architecture from the
    Binary Connect: VGG like network, in the BNN code, it's name VGG
    """
    model = Vgg_like_real_binact(**kwargs)
    # if pretrained:
    #     model_path = 'model_list/alexnet.pth.tar'
    #     pretrained_model = torch.load(model_path)
    #     model.load_state_dict(pretrained_model['state_dict'])
    return model

class RealConv2d(nn.Module): # change the name of RealConv2d
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False):
        super(RealConv2d, self).__init__()
        self.layer_type = 'RealConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        # x = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x


class VGG_like(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_like, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.bconv2 = RealConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bconv3 = RealConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bconv4 = RealConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bconv5 = RealConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bconv6 = RealConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.mpool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc7 = RealConv2d(512 * 4 * 4, 1024, Linear=True)
        self.fc8 = RealConv2d(1024, 1024, dropout=0.5, Linear=True)
        self.bn8 = nn.BatchNorm1d(1024, eps=1e-3, momentum=0.1, affine=True)
        self.dp8 = nn.Dropout()
        self.fc9 = nn.Linear(1024, num_classes)

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            10: {'lr': 5e-3},
            15: {'lr': 1e-3, 'weight_decay': 0},
            20: {'lr': 5e-4},
            25: {'lr': 1e-4}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # self.input_transform = {
        #     'train': transforms.Compose([
        #         transforms.Scale(256),
        #         transforms.RandomCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize
        #     ]),
        #     'eval': transforms.Compose([
        #         transforms.Scale(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize
        #     ])
        # }

    def forward(self, input):
        conv1out = self.conv1(input)
        bn1out = self.bn1(conv1out)
        relu1out = self.relu1(bn1out)
        bconv2out = self.bconv2(relu1out)
        mpool2out = self.mpool2(bconv2out)
        bconv3out = self.bconv3(mpool2out)
        bconv4out = self.bconv4(bconv3out)
        mpool4out = self.mpool4(bconv4out)
        bconv5out = self.bconv5(mpool4out)
        bconv6out = self.bconv6(bconv5out)
        mpool6out = self.mpool6(bconv6out)
        mpool6out = mpool6out.view(mpool6out.size(0), 512 * 4 * 4)
        fc7out = self.fc7(mpool6out)
        fc8out = self.fc8(fc7out)
        bn8out = self.bn8(fc8out)
        dp8out = self.dp8(bn8out)
        fc9out = self.fc9(dp8out)
        return fc9out#, bn1out, bconv2out, bconv3out, bconv4out, bconv5out, bconv6out, fc7out, fc8out


def vgg_like(**kwargs):
    """model architecture from the
    Binary Connect: VGG like network, in the BNN code, it's name VGG
    """
    model = VGG_like(**kwargs)
    # if pretrained:
    #     model_path = 'model_list/alexnet.pth.tar'
    #     pretrained_model = torch.load(model_path)
    #     model.load_state_dict(pretrained_model['state_dict'])
    return model
