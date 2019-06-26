# coding:utf-8
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import time, sys, random


class InceptionLayer(nn.Module):

    def __init__(self, in_channels, out_B, out_D):
        super(InceptionLayer, self).__init__()
        self.blockA = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(True)
        )

        self.blockB = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(64, out_B, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_B),
            nn.ReLU(True)
        )

        self.blockC = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96, momentum=0.1),
            nn.ReLU(True)
        )

        self.blockD = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_D, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_D),
            nn.ReLU(True)
        )

    def forward(self, x):
        x_A = self.blockA(x)
        x_B = self.blockB(x)
        x_C = self.blockC(x)
        x_D = self.blockD(x)
        return torch.cat([x_A, x_B, x_C, x_D], 1)


class Net(nn.Module):
    def __init__(self, num_classes, batch_size, frame_sample_num, input_size):
        super(Net, self).__init__()

        self.frame_sample_num = frame_sample_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.ext_feature = False

        self.Incep_first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            #nn.Dropout(p=0.3),
            nn.BatchNorm2d(192, momentum=0.1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        )

        self.IncepA = InceptionLayer(192, 64, 32)
        self.IncepB = InceptionLayer(256, 96, 64)
        self.IncepC = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=1, stride=1),
            #nn.Dropout(p=0.3),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96, momentum=0.1),
            nn.ReLU(True)
        )

        self.resA_conv = nn.Conv3d(96, 128, kernel_size=3, stride=1, padding=1)
        self.resA_BN = nn.Sequential(
            nn.BatchNorm3d(128, momentum=0.1),
            nn.ReLU(True)
        )

        self.resB_conv = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            #nn.Dropout(p=0.3),
            nn.BatchNorm3d(128, momentum=0.1),
            nn.ReLU(True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.resB_BN = nn.Sequential(
            nn.BatchNorm3d(128, momentum=0.1),
            nn.ReLU(True)
        )

        self.resC_conv = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256, momentum=0.1),
            nn.ReLU(True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.resC_2_conv = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        self.resC_BN = nn.Sequential(
            nn.BatchNorm3d(256, momentum=0.1),
            nn.ReLU(True)
        )

        self.resD_conv = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256, momentum=0.1),
            nn.ReLU(True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.resD_BN = nn.Sequential(
            nn.BatchNorm3d(256, momentum=0.1),
            nn.ReLU(True)
        )

        self.resE_conv = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(512, momentum=0.1),
            nn.ReLU(True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        )
        self.resE_2_conv = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        )
        self.resE_BN = nn.Sequential(
            nn.BatchNorm3d(512, momentum=0.1),
            nn.ReLU(True)
        )

        self.resF_conv = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.Dropout(p=0.3),
            nn.BatchNorm3d(512, momentum=0.1),
            nn.ReLU(True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        )
        self.resF_BN = nn.Sequential(
            nn.BatchNorm3d(512, momentum=0.1),
            nn.ReLU(True)
        )

        self.final_pool = nn.AvgPool3d(kernel_size=(4, 7, 7), stride=1, padding=0)
        self.pool_dropout = nn.Dropout(p=0.3)
        self.final_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        # important operation to stablize batchnorm
        x = x.view(-1, 3, self.input_size, self.input_size)

        x = self.Incep_first(x)
        x = self.IncepA(x)
        x = self.IncepB(x)
        x = self.IncepC(x)

        x = torch.transpose(x.view(-1, self.frame_sample_num, 96, 28, 28), 1, 2)
        # (batch, channel, frame_sample_num, 28, 28)

        resA = self.resA_conv(x)
        x = self.resA_BN(resA)# res3a_bn
        x = self.resB_conv(x)# res3b_2
        x += resA

        x = self.resB_BN(x)# res3b_bn
        resC = self.resC_conv(x)# res4a_2
        resC_2 = self.resC_2_conv(x)# res4a_down
        resC += resC_2# res4a
        x = self.resC_BN(resC)# res4a_bn

        x = self.resD_conv(x)# res4b_2
        x += resC# res4b
        x = self.resD_BN(x)# res4b_bn

        resE = self.resE_conv(x)
        resE_2 = self.resE_2_conv(x)
        resE += resE_2# res5a
        x = self.resE_BN(resE)# res5a_bn

        x = self.resF_conv(x)# res5b_2
        x += resE
        x = self.resF_BN(x)
        #print("pool", x.shape)
        x = self.final_pool(x).squeeze()
        x = self.pool_dropout(x)
        if self.ext_feature:
            return x

        x = self.final_layer(x)
        return x
