import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DenseBlock(nn.Module):
    def __init__(self,input_channel,growth_rate):
        super(DenseBlock,self).__init__()
        inter_channel = growth_rate*4
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.conv1 = nn.Conv2d(input_channel,inter_channel,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channel)
        self.conv2 = nn.Conv2(inter_channel,growth_rate,kernel_size=3,padding=1,bias=False)

    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x,out),1)
        return out

class Transition(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Transition,self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.conv1 = nn.Conv2d(input_channel,output_channel,kernel_size=1,bias=False)
    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out,2)
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses):
        super(DenseNet,self).__init__()

        nDenseBlocks = (depth-4) //3
        nDenseBlocks //=2

        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans3 = Transition(nChannels,nOutChannels)

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(DenseBlock(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn1(out)))
        out = out.view(out.size(0), -1)
        out = F.log_softmax(self.fc(out))
        return out