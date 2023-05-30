import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
# import resnet50
# import torchutils



class Net(nn.Module):

    def __init__(self, stride=16, n_classes=10):  #n_class set to 10 to prevent gradient explosion and overfitting,there are actually only two classes.
        super(Net, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.conv = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)
        self.stage5 = nn.Sequential(self.conv, self.bn, self.relu)

        self.n_classes = n_classes
        self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)
        self.fc = nn.Linear(2048, n_classes)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x1, x2):
        # x = torch.cat([x1,x2], 1)
        # print(x.shape)
        x = (x2-x1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # x1 = self.stage1(x1)
        # x2 = self.stage1(x2)
        # x1 = self.stage2(x1)
        # x2 = self.stage2(x2)
        # x1 = self.stage3(x1)
        # x2 = self.stage3(x2)
        # x1 = self.stage4(x1)
        # x2 = self.stage4(x2)

        # x = torch.cat((x1, x2), 1)
        # x = x2 - x1
        # x = self.stage5(x)

        x = torchutils.gap2d(x, keepdims=True)
        # print(x)
        # exit(-1)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        return x

    def train(self, mode=True):
        super(Net, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

class Net_CAM(Net):

    def __init__(self,stride=16,n_classes=10): #n_class set to 10 to prevent gradient explosion and overfitting,there are actually only two classes.
        super(Net_CAM, self).__init__(stride=stride,n_classes=n_classes)
        
    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        # print((self.classifier.weight).shape)
        # print(cams.shape)
        # exit(-1)
        cams = F.relu(cams)
        
        return x,cams,feature

class Net_CAM_Feature(Net):

    def __init__(self,stride=16,n_classes=10): #n_class set to 10 to prevent gradient explosion and overfitting,there are actually only two classes.
        super(Net_CAM_Feature, self).__init__(stride=stride,n_classes=n_classes)
        
    def forward(self, x1, x2):

        x = 2*(x2-x1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # x = self.stage4(x)
        # x1 = self.stage1(x1)
        # x2 = self.stage1(x2)
        # x1 = self.stage2(x1)
        # x2 = self.stage2(x2)
        # x1 = self.stage3(x1)
        # x2 = self.stage3(x2)
        # x1 = self.stage4(x1)
        # x2 = self.stage4(x2)

        # # x = torch.cat((x1, x2), 1)
        # x = x2 - x1
        
        feature = self.stage4(x) # bs*2048*32*32

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)
        cams = cams/(F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2)*feature.unsqueeze(1) 
        cams_feature = cams_feature.view(cams_feature.size(0),cams_feature.size(1),cams_feature.size(2),-1)
        cams_feature = torch.mean(cams_feature,-1)
        
        return x,cams_feature,cams

class CAM(Net):

    def __init__(self, stride=16,n_classes=10): #n_class set to 10 to prevent gradient explosion and overfitting,there are actually only two classes.
        super(CAM, self).__init__(stride=stride,n_classes=n_classes)

    def forward(self, x1, x2, separate=False):
        x = (x2-x1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # x = self.stage1(x)
        # x1 = self.stage1(x1)
        # x2 = self.stage1(x2)
        # x1 = self.stage2(x1)
        # x2 = self.stage2(x2)
        # x1 = self.stage3(x1)
        # x2 = self.stage3(x2)
        # x1 = self.stage4(x1)
        # x2 = self.stage4(x2)

        # # x = torch.cat((x1, x2), 1)
        # x = x2 - x1
        # x = self.stage5(x)

        x = F.conv2d(x, self.classifier.weight)
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)
        return x

    def forward1(self, x1, x2, weight, separate=False):
        x = (x2-x1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weight)
        
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward2(self, x1, x2, weight, separate=False):
        
        x = (x2-x1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # x1 = self.stage1(x1)
        # x2 = self.stage1(x2)
        # x1 = self.stage2(x1)
        # x2 = self.stage2(x2)
        # x1 = self.stage3(x1)
        # x2 = self.stage3(x2)
        # x1 = self.stage4(x1)
        # x2 = self.stage4(x2)

        # # x = torch.cat((x1, x2), 1)
        # x = x2 - x1
        # x = self.stage5(x)

        x = F.conv2d(x, weight*self.classifier.weight)
        
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)
        return x

class Class_Predictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(representation_size, num_classes, 1, bias=False)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = x.reshape(batch_size,self.num_classes,-1) 
        mask = label>0 
        feature_list = [x[i][mask[i]] for i in range(batch_size)] 
        prediction = [self.classifier(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]

        loss = 0
        acc = 0
        num = 0
        for logit,label in zip(prediction, labels):
            if label.shape[0] == 0:
                continue
            loss_ce= F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1)==label.view(-1)).sum().float()
            num += label.size(0)
        
            
        return loss/batch_size, acc/num
    




if __name__ == '__main__':
    model = CAM()
    # in_batch, inchannel, in_h, in_w = 2, 3, 256, 256
    in_batch, inchannel, in_h, in_w = 8, 3, 256, 256
    # in_batch, inchannel, in_h, in_w = 1, 3, 300, 300
    x1 = torch.randn(in_batch, inchannel, in_h, in_w)
    
    # in_batch, inchannel, in_h, in_w = 8, 64, 64, 64
    # x2 = torch.randn(in_batch, inchannel, in_h, in_w)
    # print(x1.shape)
    # print(x2.shape)
    # v1, v2 = model(x1, x2)
    out = model(x1)
    # out = model(x1)
    print(out.shape)
    # print(v2.shape)
