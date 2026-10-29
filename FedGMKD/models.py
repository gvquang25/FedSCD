from PIL import Image
from os.path import join
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models

       
        
class ResNet18(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ResNet18, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes  
        self.feature_extractor = models.resnet18(num_classes=self.code_length)
        self.classifier =  nn.Sequential(
                                nn.Linear(self.code_length, self.num_classes))
    def forward(self,x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z,p
    