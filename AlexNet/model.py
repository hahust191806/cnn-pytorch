import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), (4, 4), (2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)),
            
            nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2)), 
            
            nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(True), 
            nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(True), 
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            nn.MaxPool2d((3, 3), (2, 2))
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(256*6*6, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5), 
            nn.Linear(4096, 4096), 
            nn.ReLU(True), 
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x): 
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out 


if __name__ == "__main__":
    model = AlexNet(10)
    print(model)