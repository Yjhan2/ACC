
import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import functools

from torch.nn import BatchNorm2d as BatchNorm2d
from torch.nn import BatchNorm1d as BatchNorm1d

def conv2d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False),
        BatchNorm2d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)

def conv1d(in_channel, out_channel, kernel_size):
    layers = [
        nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size // 2, bias=False),
        BatchNorm1d(out_channel),
        nn.ReLU(),
    ]

    return nn.Sequential(*layers)

class ACCAtention(nn.Module):
    def __init__(self, feat_in=512, feat_out=256, num_classes=20, size=[384//16,384//16], head = 1 ):
        super(ACCAtention, self).__init__()
        h,w = size[0],size[1]
        self.gamma = Parameter(torch.ones(1))
        self.beta = Parameter(torch.ones(1))
        self.rowatt = RowAttention(feat_in,feat_in)
        self.colatt = ColAttention(feat_in,feat_in)
        self.h_conv = nn.Conv2d(feat_in,num_classes)
        self.rowpool = nn.AdaptiveAvgPool2d((h,1))
        self.w_conv = nn.Conv2d(feat_in,num_classes)
        self.colpool = nn.AdaptiveAvgPool2d((1,w))
        self.ccatt = CCAttention(feat_in,feat_out)

    def forward(self,fea):
        fea_h = self.rowatt(fea)
        fea_w = self.colatt(fea)
        fea_hp = self.h_conv(fea_h).squeeze(3)
        fea_wp = self.w_conv(fea_w).squeeze(2)
        fea = self.gamma*fea_h + self.beta*fea_w
        fea = self.ccatt(fea)
        return fea, fea_hp, fea_wp


def INF(B,H,W):
    '''
    生成(B*W,H,H)大小的对角线为inf的三维矩阵
    Parameters
    ----------
    B: batch
    H: height
    W: width
    '''
    return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CCAttention(nn.Module):

    def __init__(self, in_dim, device):
        '''
        Parameters
        ----------
        in_dim : int
            channels of input
        '''
        super(CCAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1)).to(device)
        self.device = device

    def forward(self, x):
        m_batchsize, _, height, width = x.size()

        proj_query = self.query_conv(x)  # size = (b,c2,h,w), c1 = in_dim, c2 = c1 // 8

        # size = (b*w, h, c2)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)

        # size = (b*h, w, c2)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)

        proj_key = self.key_conv(x)  # size = (b,c2,h,w)

        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1,
                                                                    width)  # size = (b*w,c2,h)

        proj_value = self.value_conv(x)  # size = (b,c1,h,w)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1,
                                                                        height)  # size = (b*w,c1,h)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1,
                                                                        width)  # size = (b*h,c1,w)

        # size = (b*w, h,h) ,其中[:,i,j]表示Q所有W的第Hi行的所有通道值与K上所有W的第Hj列的所有通道值的向量乘积
        energy_H = torch.bmm(proj_query_H, proj_key_H)

        # size = (b,h,w,h) #这里为什么加 INF并没有理解
        energy_H = (energy_H + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0,
                                                                                                                      2,
                                                                                                                      1,
                                                                                                                      3)

        # size = (b*h,w,w),其中[:,i,j]表示Q所有H的第Wi行的所有通道值与K上所有H的第Wj列的所有通道值的向量乘积
        energy_W = torch.bmm(proj_query_W, proj_key_W)
        energy_W = energy_W.view(m_batchsize, height, width, width)  # size = (b,h,w,w)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))  # size = (b,h,w,h+w) #softmax归一化
        # concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
                                                                                 height)  # size = (b*w,h,h)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width,
                                                                          width)  # size = (b*h,w,w)

        # size = (b*w,c1,h) #[:,i,j]表示V所有W的第Ci行通道上的所有H 与att_H的所有W的第Hj列的h权重的乘积
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1))
        out_H = out_H.view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)  # size = (b,c1,h,w)

        # size = (b*h,c1,w) #[:,i,j]表示V所有H的第Ci行通道上的所有W 与att_W的所有H的第Wj列的W权重的乘积
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1))
        out_W = out_W.view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)  # size = (b,c1,h,w)
        # print(out_H.size(),out_W.size())

        return self.gamma * (out_H + out_W) + x



class RowAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim, device):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.device = device

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1)).to(self.device)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h,c2,w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h, c1,w)

        # size = (b*h,w,w) [:,i,j] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有h的第 Wj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        row_attn = torch.bmm(Q, K)
        ########
        # 此时的 row_atten的[:,i,0:w] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有行的 所有列(0:w)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:w）逐个位置的值的乘积，得到行attn
        ########

        # 对row_attn进行softmax
        row_attn = self.softmax(row_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*h,c1,w) 这里先需要对row_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 row_attn的行的乘积，即求权重和
        out = torch.bmm(V, row_attn.permute(0, 2, 1))

        # size = (b,c1,h,2)
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)

        out = self.gamma * out + x

        return out

class ColAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim, device):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.device = device

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1)).to(self.device)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h)

        # size = (b*w,h,h) [:,i,j] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的第 Hj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        col_attn = torch.bmm(Q, K)
        ########
        # 此时的 col_atten的[:,i,0:w] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的 所有列(0:h)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:h）逐个位置的值的乘积，得到列attn
        ########

        # 对row_attn进行softmax
        col_attn = self.softmax(col_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*w,c1,h) 这里先需要对col_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 col_attn的行的乘积，即求权重和
        out = torch.bmm(V, col_attn.permute(0, 2, 1))

        # size = (b,c1,h,w)
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)

        out = self.gamma * out + x

        return out


class CDGAttention(nn.Module):
    def  __init__(self, feat_in=512, feat_out=256, num_classes=20, size=[384//16,384//16], kernel_size =7 ):
        super(CDGAttention, self).__init__()   
        h,w = size[0],size[1]
        kSize = kernel_size
        self.gamma = Parameter(torch.ones(1))
        self.beta = Parameter(torch.ones(1))
        self.rowpool = nn.AdaptiveAvgPool2d((h,1))
        self.colpool = nn.AdaptiveAvgPool2d((1,w))
        self.conv_hgt1 =conv1d(feat_in,feat_out,3)
        self.conv_hgt2 =conv1d(feat_in,feat_out,3)
        self.conv_hwPred1 = nn.Sequential(
            nn.Conv1d(feat_out,num_classes,3,stride=1,padding=1,bias=True),
            nn.Sigmoid(),   
        )
        self.conv_hwPred2 = nn.Sequential(
            nn.Conv1d(feat_out,num_classes,3,stride=1,padding=1,bias=True),
            nn.Sigmoid(),                                                            
         )         
        self.conv_upDim1 = nn.Sequential(
            nn.Conv1d(feat_out,feat_in,kSize,stride=1,padding=kSize//2,bias=True),  
            nn.Sigmoid(),                                                                              
        )
        self.conv_upDim2 = nn.Sequential( 
            nn.Conv1d(feat_out,feat_in,kSize,stride=1,padding=kSize//2,bias=True),  
            nn.Sigmoid(),                                                                            
        )
        self.cmbFea = conv2d( feat_in*3,feat_in,3)        
    def forward(self,fea):
        n,c,h,w = fea.size()       
        fea_h = self.rowpool(fea).squeeze(3)      #先转换为n,c,1,h,再转换为n,c,h
        fea_w = self.colpool(fea).squeeze(2)      #先转换为n,c,w,1,再转换为n,c,w
        fea_h = self.conv_hgt1(fea_h)             #n,c,h，c=256
        fea_w = self.conv_hgt2(fea_w) 
        #===========================================================               
        fea_hp = self.conv_hwPred1(fea_h)            #n,class_num,h
        fea_wp = self.conv_hwPred2(fea_w)            #n,class_num,w 
        #===========================================================
        fea_h = self.conv_upDim1(fea_h)                 #n,c,h.c=512
        fea_w = self.conv_upDim2(fea_w)                 #n,c,w.c=512
        fea_hup = fea_h.unsqueeze(3)                    #扩展第3维得到n,c,h,1
        fea_wup = fea_w.unsqueeze(2)                    #扩展第2维得到n,c,1,w
        fea_hup = F.interpolate( fea_hup, (h,w), mode='bilinear', align_corners= True ) #n,c,h,w
        fea_wup = F.interpolate( fea_wup, (h,w), mode='bilinear', align_corners= True ) #n,c,h,w       
        fea_hw = self.beta*fea_wup + self.gamma*fea_hup        
        fea_hw_aug = fea * fea_hw        
        #===============================================================      
        fea = torch.cat([fea, fea_hw_aug, fea_hw], dim = 1 )
        fea = self.cmbFea( fea )       
        return fea, fea_hp, fea_wp
        
class C2CAttention(nn.Module):
    def  __init__(self, in_fea, out_fea, num_class ):
        super(C2CAttention, self).__init__()
        self.in_fea = in_fea
        self.out_fea = out_fea
        self.num_class = num_class
        self.gamma = Parameter(torch.ones(1))
        self.beta = Parameter(torch.ones(1))
        self.bias1 = Parameter( torch.FloatTensor( num_class, num_class ))
        self.bias2 = Parameter( torch.FloatTensor( num_class, num_class ))
        self.convDwn1 = conv2d( in_fea, out_fea, 1 )
        self.convDwn2 = conv2d( in_fea, out_fea, 1 )
        self.convUp1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            conv2d( num_class, out_fea, 1 ),
            nn.Conv2d(out_fea,in_fea,1,stride=1,padding=0,bias=True),                       
        )
        self.toClass = nn.Sequential(
            nn.Conv2d( out_fea, num_class, 1, stride=1, padding = 0, bias = True ),          
        )       
        self.convUp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            conv2d( num_class, out_fea, 1 ),
            nn.Conv2d(out_fea,in_fea,1,stride=1,padding=0,bias=True),                        
        )        
        self.fea_fuse = conv2d( in_fea*2, in_fea, 1 )
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
    def reset_parameters(self):        
        torch.nn.init.xavier_uniform_(self.bias1)  
        torch.nn.init.xavier_uniform_(self.bias2)   
    def forward(self,input_fea):  
        n, c, h, w = input_fea.size()        
        fea_ha = self.convDwn1( input_fea )
        fea_wa = self.convDwn2( input_fea )
        cls_ha = self.toClass( fea_ha )
        cls_ha = F.softmax(cls_ha, dim=1)
        cls_wa = self.toClass( fea_wa )
        cls_wa = F.softmax(cls_wa, dim=1)
        cls_ha = cls_ha.view( n, self.num_class, h*w )
        cls_wa = cls_wa.view( n, self.num_class, h*w )
        cch = F.relu(torch.matmul( cls_ha, cls_ha.transpose( 1, 2 ) ))  #class*class
        cch = cch 
        cch = self.sigmoid( cch ) + self.bias1                           
        ccw = F.relu(torch.matmul( cls_wa, cls_wa. transpose( 1, 2 ) )) #class*class
        ccw = ccw 
        ccw = self.sigmoid( ccw )+ self.bias2                            
        cls_ha = torch.matmul( cls_ha.transpose(1,2), cch.transpose(1,2) )
        cls_ha = cls_ha.transpose( 1,2).contiguous().view( n, self.num_class, h, w )
        cls_wa = torch.matmul( cls_wa.transpose(1,2), ccw.transpose(1,2) )
        cls_wa = cls_wa.transpose(1,2).contiguous().view( n, self.num_class, h, w )        
        fea_ha = self.convUp1( cls_ha )
        fea_wa = self.convUp2( cls_wa )
        fea_hwa = self.gamma*fea_ha + self.beta*fea_wa
        fea_hwa_aug = input_fea * fea_hwa                   #*
        fea_fuse = torch.cat( [fea_hwa_aug, input_fea], dim = 1 )
        fea_fuse = self.fea_fuse( fea_fuse )
        return fea_fuse, cch, ccw  

class StatisticAttention(nn.Module):
    def  __init__(self,fea_in, fea_out, num_classes ):
       super(StatisticAttention, self).__init__()
    #    self.gamma = Parameter(torch.ones(1))
       self.conv_1 = conv2d( fea_in, fea_in//2, 1)      #kernel size 3
       self.conv_2 = conv2d( fea_in//2, num_classes, 3 )
       self.conv_pred = nn.Sequential(
           nn.Conv2d( num_classes, 1, 3, stride=1, padding=1, bias=True),   #kernel size 1
           nn.Sigmoid()
       )
       self.conv_fuse = conv2d( fea_in * 2, fea_out, 3 )
    def forward(self,fea):
        fea_att = self.conv_1( fea )
        fea_cls = self.conv_2( fea_att )
        fea_stat = self.conv_pred( fea_cls )     
        fea_aug = fea * ( 1 - fea_stat ) 
        fea_fuse = torch.cat( [fea, fea_aug], dim = 1 )
        fea_res = self.conv_fuse( fea_fuse )
        return fea_res, fea_stat        
        
class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 7, 11), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center
        
class PCM(Module):
    def __init__(self, feat_channels=[256,1024]):
        super().__init__()
        feat1, feat2 = feat_channels
        self.conv_x2 = conv2d( feat1, 256, 1 )
        self.conv_x4 = conv2d( feat2, 256, 1 )
        self.conv_cmb = conv2d( 256+256+3, 256, 1 )
        self.softmax = Softmax(dim=-1)
        self.psp = PSPModule()
        self.addCAM = conv2d( 512, 256, 1)    
    def forward(self, xOrg, stg2, stg4, cam ):
        n,c,h,w = stg2.size()
        stg2 = self.conv_x2( stg2 )
        stg4 = self.conv_x4( stg4 )
        stg4 = F.interpolate( stg4, (h,w), mode='bilinear', align_corners= True)
        stg0 = F.interpolate( xOrg, (h,w), mode='bilinear', align_corners= True)
        stgSum = torch.cat([stg0,stg2,stg4],dim=1)
        stgSum = self.conv_cmb( stgSum )
        stgPool = self.psp( stgSum )                            #(N,c,s)
        stgSum = stgSum.view( n, -1, h*w ).transpose(1,2)       #(N,h*w,c)
        stg_aff = torch.matmul( stgSum, stgPool ) #(N,h*w,c)*(N,c,s)=(N,h*w,s)
        stg_aff = ( c ** -0.5 ) * stg_aff
        stg_aff = F.softmax( stg_aff, dim = -1 ) #(N,h*w,s)
        with torch.no_grad():
            cam_d = F.relu( cam.detach() ) 
        cam_d = F.interpolate( cam_d, (h,w), mode='bilinear', align_corners= True)
        cam_pool = self.psp( cam_d ).transpose(1,2) #(N,s,c)
        cam_rv = torch.matmul( stg_aff, cam_pool ).transpose(1,2)
        cam_rv=cam_rv.view(n, -1, h, w )
        out = torch.cat([cam, cam_rv], dim=1 )
        out = self.addCAM( out )
        return out
        
class GCM(Module):
    def __init__(self, feat_channels=512):
        super().__init__()

        chHig = feat_channels
        self.gamma = Parameter(torch.ones(1))
        self.higC = conv2d( chHig, 256, 3 )
        self.coe = nn.Sequential(                  
            conv2d( 256, 256, 3 ),
            nn.AdaptiveAvgPool2d((1,1)) 
        )       

    def forward(self, fea ):
        n,_,h, w = fea.size() 
        stgHig = self.higC( fea )
        coeHig = self.coe( stgHig )      
        sim = stgHig - coeHig
        # print( sim.size() )
        simDis = torch.norm( sim, 2, 1, keepdim = True )
        # print( simDis.size() )
        simDimMin = simDis.view( n, -1 )
        simDisMin = torch.min( simDimMin, 1, keepdim = True )[0]        
        # print( simDisMin.size() )
        simDis = simDis.view( n, -1 )
        weightHig = torch.exp( -( simDis - simDisMin ) / 5 )
        weightHig = weightHig.view(n, -1, h, w )
        upFea = F.interpolate( coeHig, (h,w), mode='bilinear', align_corners=True)
        upFea = upFea * weightHig
        stgHig = stgHig + self.gamma * upFea

        return weightHig, stgHig

class LCM(Module):
    def __init__(self, feat_channels=[256, 256, 512]):
        super().__init__()
        
        chHig, chLow1, chLow2 = feat_channels
        self.beta = Parameter(torch.ones(1)) 
        self.lowC1 = conv2d( chLow1, 48,3)
        self.lowC2 = conv2d( chLow2,128,3)
        self.cat1 = conv2d( 256+48, 256, 1 )
        self.cat2 = conv2d( 256+128, 256, 1 )     

    def forward(self, feaHig, feaCeo, feaLow1, feaLow2 ):        
        n,c,h,w = feaLow1.size()
        stgHig = F.interpolate( feaHig, (h,w), mode='bilinear', align_corners=True)
        weightLow = F.interpolate( feaCeo, (h,w), mode='bilinear', align_corners=True )
        coeLow = 1 - weightLow  
        stgLow1 = self.lowC1(feaLow1)
        stgLow2 = self.lowC2(feaLow2)   
        stgLow2 = F.interpolate( stgLow2, (h,w), mode='bilinear', align_corners=True )

        stgLow1 = self.beta * coeLow * stgLow1
        stgCat = torch.cat( [stgHig, stgLow1], dim = 1 )
        stgCat = self.cat1( stgCat )
        stgLow2 = self.beta * coeLow * stgLow2
        stgCat = torch.cat( [stgCat, stgLow2], dim = 1 )
        stgCat = self.cat2( stgCat )       
        return stgCat
