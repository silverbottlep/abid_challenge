import torch
import torch.nn as nn

class siamese_resnet(nn.Module):
  def __init__(self, resnet):
    super(siamese_resnet, self).__init__()
    self.resnet = resnet
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(1024, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, 1),
      nn.Sigmoid()
    )

  def forward(self, x1, x2):
    x1 = self.resnet.conv1(x1)
    x1 = self.resnet.bn1(x1)
    x1 = self.resnet.relu(x1)
    x1 = self.resnet.maxpool(x1)

    x1 = self.resnet.layer1(x1)
    x1 = self.resnet.layer2(x1)
    x1 = self.resnet.layer3(x1)
    x1 = self.resnet.layer4(x1)
        
    x1 = self.resnet.avgpool(x1)
    x1 = x1.view(x1.size(0), -1)
        
    x2 = self.resnet.conv1(x2)
    x2 = self.resnet.bn1(x2)
    x2 = self.resnet.relu(x2)
    x2 = self.resnet.maxpool(x2)

    x2 = self.resnet.layer1(x2)
    x2 = self.resnet.layer2(x2)
    x2 = self.resnet.layer3(x2)
    x2 = self.resnet.layer4(x2)
        
    x2 = self.resnet.avgpool(x2)
    x2 = x2.view(x2.size(0), -1)
    
    concat = torch.cat([x1,x2],1)

    p = self.classifier(concat)
    return p

class siamese_vgg(nn.Module):
  def __init__(self, vgg):
    super(siamese_vgg, self).__init__()
    self.vgg = vgg
    self.avgpool = nn.AvgPool2d(7)
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(1024, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, 1),
      nn.Sigmoid()
    )

  def forward(self, x1, x2):
    x1 = self.vgg.features(x1)
    x1 = self.avgpool(x1)
    x1 = x1.view(x1.size(0), -1)
        
    x2 = self.vgg.features(x2)
    x2 = self.avgpool(x2)
    x2 = x2.view(x2.size(0), -1)
    
    concat = torch.cat([x1,x2],1)

    p = self.classifier(concat)
    return p
