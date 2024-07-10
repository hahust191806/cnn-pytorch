import torch.nn as nn # this module contains predefined layers and modules in pytorch 
import torch.nn.functional as F # this module contains functinal operations of pytorch
import numpy as np  

"""
    Output_size of convolutional layer: (w - k + 2p)/ s
"""

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        """
        One forward pass through the network 
        
        Args: 
            x: input
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x)) # used to reshape tensor   
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    
    def num_flat_features(self, x):
        """
        Get the number of features in a batch of tensors x
        """
        size = x.size()[1:]
        return np.prod(size)
    
    
if __name__ == "__main__":
    lenet_model = LeNet()
    print(lenet_model)