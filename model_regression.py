import torch
import torch.nn as nn
class CountRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # 384,384->44,44->1
        self.conv1 = nn.Conv2d(1, 16, 3) 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3) 
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  
        
        self.conv3 = nn.Conv2d(32, 64, 3) 
        self.bn3 = nn.BatchNorm2d(64)  
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  
        
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.bn6 = nn.BatchNorm2d(512)  
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 44 * 44, 1024) 
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)   


    def forward(self, x):
        x = self.pool1(torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))))
        x = self.pool2(torch.relu(self.bn4(self.conv4(torch.relu(self.bn3(self.conv3(x)))))))
        x = self.pool3(torch.relu(self.bn6(self.conv6(torch.relu(self.bn5(self.conv5(x)))))))
        #print(x.shape)
        x = x.view(-1, 512 * 44 * 44)  
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.shape)
        return x

def regression_model():
    return CountRegression()