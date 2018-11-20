import torch
import numpy as np
import torch.nn as nn

class WeightedL1_Loss(nn.Module):
    def __init__(self):
        super(WeightedL1_Loss, self).__init__()
        return

    def forward(self, weight, data1, data2):
        diff = torch.abs(data1 - data2)
        output = torch.mul(weight, diff)
        return output.sum()