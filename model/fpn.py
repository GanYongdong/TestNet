import torch.nn as nn
import torch.nn.functional as F
import math

class FPN(nn.Module):
    '''resnet50,101,152'''
    def __init__(self,features=256,use_p5=True):
        super(FPN,self).__init__()
        self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.conv_5= nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4= nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3= nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5 = use_p5
    def upsamplelike(self,inputs):
        src,target=inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3])),