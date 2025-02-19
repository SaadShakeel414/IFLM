import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt

class HCAM(nn.Module):
    def __init__(self,in_channels,r=4):
        super(HCAM,self).__init__()

        self.r = r
        self.conv1 = nn.Conv2d(in_channels,in_channels//r,kernel_size=1,bias=False)
        self.convDW = nn.Conv2d(in_channels//r, in_channels//r, kernel_size=3, stride=1, padding=1, groups=in_channels//r, bias=False)
        self.convPW = nn.Conv2d(in_channels//r,in_channels,kernel_size=1,bias=False)
        self.GL_CA = Local_Global_Channel_Att(in_channels,r)

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x_dw = self.convDW(x)
        x_pw = self.convPW(x_dw)
        CA_x = self.GL_CA(x_pw)

        return CA_x + residual

class IFLM_Spatial_Channel(nn.Module):
    def __init__(self, channels=512, reduction=4):
        super(IFLM_Spatial_Channel, self).__init__()

        self.conva1 = nn.Conv2d(channels, 128, (3, 3), dilation=2, padding=2)
        self.conva2 = nn.Conv2d(channels, 128, (3, 3), dilation=3, padding=3)
        self.conva3 = nn.Conv2d(channels, 128, (3, 3), dilation=5, padding=5)
        self.conva4 = nn.Conv2d(channels, 128, (3, 3), dilation=7, padding=7)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False)
        self.dense2 = nn.Conv2d(256,512,kernel_size=1, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CCIFL = Cross_Channel_Interaction(channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, w, h = x.size()

        ################First branch#####################
        a1 = self.relu(self.conva1(x)).view(b, -1, w * h).permute(0, 2, 1)
        a2 = self.relu(self.conva2(x)).view(b, -1, w * h)
        a3 = self.relu(self.conva3(x)).view(b, -1, w * h).permute(0, 2, 1)
        a4 = self.relu(self.conva4(x)).view(b, -1, w * h)

        energy12 = torch.bmm(a1, a2)
        attention12 = self.softmax(energy12)

        attention_output1 = torch.bmm(a4, attention12)
        attention_output1 = attention_output1.view(b, -1, w, h)
        attention_output1 = self.dense(attention_output1)  ####[B,512,7,7]

        ###############Second branch###################
        energy34 = torch.bmm(a3, a4)
        attention34 = self.softmax(energy34)

        attention_output2 = torch.bmm(a2, attention34)
        attention_output2 = attention_output2.view(b, -1, w, h)
        attention_output2 = self.dense(attention_output2)

        #####################Cross-Channel Interaction#####################
        C_Att_1 = self.CCIFL(attention_output1)
        C_Att_1 = self.dense2(C_Att_1)
        C_Att_2 = self.CCIFL(attention_output2)
        C_Att_2 = self.dense2(C_Att_2)

        Att_output = (C_Att_1 + C_Att_2) + x

        return Att_output

class SFEB(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,dilation):
        super(SFEB,self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=False),
                                  nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=False))

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):

        fea = self.relu(self.conv(x))

        return fea

class SCFIM(nn.Module):
    def __init__(self, channels, r):
        super(SCFIM, self).__init__()

        self.LF1 = nn.Sequential(nn.Conv2d(channels, channels//r, kernel_size=3, stride=1, padding=2, dilation=2, groups=channels//r, bias=False),
                                 nn.Conv2d(channels//r, channels// r, kernel_size=1, bias = False)
                                 )

        self.LF2 = nn.Sequential(nn.Conv2d(channels, channels // r, kernel_size=3, stride=1, padding=3, dilation=3, groups=channels // r, bias = False),
                                 nn.Conv2d(channels // r, channels// r, kernel_size=1, bias=False)
                                 )

        self.GF1 = nn.Sequential(nn.Conv2d(channels, channels // r, kernel_size=3, stride=1, padding=5, dilation=5, groups=channels // r, bias = False),
                                 nn.Conv2d(channels // r, channels // r, kernel_size=1, bias=False)
                                 )

        self.GF2 = nn.Sequential(nn.Conv2d(channels, channels // r, kernel_size=3, stride=1, padding=7, dilation=7, groups=channels // r, bias = False),
                                 nn.Conv2d(channels // r, channels // r, kernel_size=1, bias=False)
                                 )


        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False)
        self.dense2 = nn.Conv2d(256,512,kernel_size=1, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CCIFL = Cross_Channel_Interaction(channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, w, h = x.size()

        ################First branch#####################
        a1 = self.relu(self.LF1(x)).view(b, -1, w * h).permute(0, 2, 1)
        a2 = self.relu(self.LF2(x)).view(b, -1, w * h)
        a3 = self.relu(self.GF1(x)).view(b, -1, w * h).permute(0, 2, 1)
        a4 = self.relu(self.GF2(x)).view(b, -1, w * h)

        energy12 = torch.bmm(a1, a2)
        attention12 = self.softmax(energy12)

        attention_output1 = torch.bmm(a4, attention12)
        attention_output1 = attention_output1.view(b, -1, w, h)
        attention_output1 = self.dense(attention_output1)  ####[B,512,7,7]

        ###############Second branch###################
        energy34 = torch.bmm(a3, a4)
        attention34 = self.softmax(energy34)

        attention_output2 = torch.bmm(a2, attention34)
        attention_output2 = attention_output2.view(b, -1, w, h)
        attention_output2 = self.dense(attention_output2)

        #####################Cross-Channel Interaction#####################
        C_Att_1 = self.CCIFL(attention_output1)
        C_Att_1 = self.dense2(C_Att_1)
        C_Att_2 = self.CCIFL(attention_output2)
        C_Att_2 = self.dense2(C_Att_2)

        Att_output = (C_Att_1 + C_Att_2) + x

        return Att_output


class SFIM(nn.Module):
    def __init__(self, channels, r):
        super(SFIM, self).__init__()

        self.LF1 = nn.Sequential(nn.Conv2d(channels, channels//r, kernel_size=3, stride=1, padding=2, dilation=2, groups=channels//r, bias=False),
                                 nn.Conv2d(channels//r, channels// r, kernel_size=1, bias = False)
                                 )

        self.LF2 = nn.Sequential(nn.Conv2d(channels, channels // r, kernel_size=3, stride=1, padding=3, dilation=3, groups=channels // r, bias = False),
                                 nn.Conv2d(channels // r, channels// r, kernel_size=1, bias=False)
                                 )

        self.GF1 = nn.Sequential(nn.Conv2d(channels, channels // r, kernel_size=3, stride=1, padding=5, dilation=5, groups=channels // r, bias = False),
                                 nn.Conv2d(channels // r, channels // r, kernel_size=1, bias=False)
                                 )

        self.GF2 = nn.Sequential(nn.Conv2d(channels, channels // r, kernel_size=3, stride=1, padding=7, dilation=7, groups=channels // r, bias = False),
                                 nn.Conv2d(channels // r, channels // r, kernel_size=1, bias=False)
                                 )

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False)
        self.dense2 = nn.Conv2d(256,512,kernel_size=1, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CCIFL = Cross_Channel_Interaction(channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b, c, w, h = x.size()

        ################First branch#####################
        a1 = self.relu(self.LF1(x)).view(b, -1, w * h).permute(0, 2, 1)
        a2 = self.relu(self.LF2(x)).view(b, -1, w * h)
        a3 = self.relu(self.GF1(x)).view(b, -1, w * h).permute(0, 2, 1)
        a4 = self.relu(self.GF2(x)).view(b, -1, w * h)

        energy12 = torch.bmm(a1, a2)
        attention12 = self.softmax(energy12)

        attention_output1 = torch.bmm(a4, attention12)
        attention_output1 = attention_output1.view(b, -1, w, h)
        attention_output1 = self.dense(attention_output1)  ####[B,512,7,7]

        ###############Second branch###################
        energy34 = torch.bmm(a3, a4)
        attention34 = self.softmax(energy34)

        attention_output2 = torch.bmm(a2, attention34)
        attention_output2 = attention_output2.view(b, -1, w, h)
        attention_output2 = self.dense(attention_output2)


        Att_output = (attention_output1 + attention_output2) + x

        return Att_output

class CFIM(nn.Module):
    def __init__(self, channels):
        super(CFIM, self).__init__()

        self.dense2 = nn.Conv2d(256,512,kernel_size=1, padding=0, bias=False)
        self.CCIFL = Cross_Channel_Interaction(channels)

    def forward(self, x):
        b, c, w, h = x.size()

        ################First branch#####################

        #####################Cross-Channel Interaction#####################
        C_Att_1 = self.CCIFL(x)
        C_Att_1 = self.dense2(C_Att_1)
        # C_Att_2 = self.CCIFL(attention_output2)
        # C_Att_2 = self.dense2(C_Att_2)
        #
        Att_output = C_Att_1 + x

        return Att_output

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)

    def forward(self, inputs):
        return self.op(inputs)

class Local_Global_Channel_Att(nn.Module):
    def __init__(self, in_channels, r):
        super(Local_Global_Channel_Att, self).__init__()

        self.LA = nn.Sequential(nn.Conv2d(in_channels, in_channels // r, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(in_channels // r),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // r, in_channels, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(in_channels))

        self.GA = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_channels, in_channels // r, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(in_channels // r),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // r, in_channels, kernel_size=1, stride=1, padding=0),
                                nn.BatchNorm2d(in_channels))

        self.LA1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // r, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_channels // r),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels // r, in_channels, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_channels))

        self.GA1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, in_channels // r, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_channels // r),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels // r, in_channels, kernel_size=1, stride=1, padding=0),
                                 nn.BatchNorm2d(in_channels))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        GCA = self.GA(x)
        GCA_w = self.sigmoid(GCA)
        LCA = self.LA(x)
        LCA_w = self.sigmoid(LCA)
        GL_CA = GCA_w + LCA_w

        return GL_CA + x

###################################################################################################
#########################################################################################################################
class Cross_Channel_Interaction(nn.Module):
    def __init__(self,in_channels):
        super(Cross_Channel_Interaction,self).__init__()

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels,in_channels//2,kernel_size=1, padding=0, bias=True)
        # self.conv1 = nn.Conv1d(in_channels,in_channels,1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        self.Q1 = nn.Conv1d(1, 1, kernel_size=1)
        self.K1 = nn.Conv1d(1, 1, kernel_size=1)
        self.V1 = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self,x):
        # b,c,h,w = x.size()
        Xc = self.GAP(x)   ###[64,512,1,1]
        bc,cc,hc,wc = Xc.size()

        Xc_q = self.conv(Xc).view(bc,-1,hc*wc).permute(0,2,1) ###[64,1,256]
        Xc_k = self.conv(Xc).view(bc,-1,hc*wc)  ###([64, 256, 1])
        Xc_v = self.conv(Xc).view(bc,-1,hc*wc)  ##3([64, 256, 1, 1])

        energy_qk = torch.bmm(Xc_q,Xc_k)  ####[64,1,1]
        att_qk = self.softmax(energy_qk)  ####[64,1,1]

        att_qkv = torch.bmm(Xc_v,att_qk)
        att_qkv = att_qkv.view(bc,-1,hc,wc)

        return att_qkv
#########################################################################################################################

