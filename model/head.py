import torch.nn as nn
import torch
import math

class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__() # 包含父类的init
        self.scale=nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class ClsCntRegHead(nn.Module):
    def __init__(self, in_channel, class_num, GN=True, cnt_on_reg=True, prior=0.01):
        super(ClsCntRegHead,self).__init__()
        self.prior=prior
        self.class_num=class_num
        self.cnt_on_reg=cnt_on_reg

        cls_branch=[] # 创建list
        reg_branch=[]

        for i in range(4):
            cls_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True)) #list末尾添加
            if GN:
                cls_branch.append(nn.GroupNorm(32,in_channel))
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32,in_channel))
            reg_branch.append(nn.ReLU(True))

        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)

        self.cls_logits=nn.Conv2d(in_channel,class_num,kernel_size=3,padding=1)
        self.cnt_logits=nn.Conv2d(in_channel,1,kernel_size=3,padding=1)
        self.reg_pred=nn.Conv2d(in_channel,4,kernel_size=3,padding=1)

        self.apply(self.init_conv_RandomNormal) # ???

        nn.init.constant_(self.cls_logits.bias,-math.log((1-prior)/prior)) # ???这个计算log0.99有啥用
        self.scale_exp=nn.ModuleList([ScaleExp(1.0) for _ in range(5)]) # ???这个scale——exp就不知道干嘛的

    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module,nn.Conv2d):
            nn.init.normal_(module.weight,std=std) # ???这个正态分布，以哪个元素为中心呀，感觉不对劲

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self,input):
        '''inputs:[p3~p7]'''
        cls_logits=[] # 这个和成员变量self.clc_logits是不同的哦
        cnt_logits=[]
        reg_preds=[]
        for index,P in enumerate(inputs):  # 金字塔有5个特征层级，分别进行如下操作
            cls_conv_out=self.cls_conv(P)
            reg_conv_out=self.reg_conv(P)

            cls_logits.append(self.cls_logits(cls_conv_out))
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        # return cls_logits,cnt_logits,reg_preds
        return 
