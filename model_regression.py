import torch
import torch.nn as nn
class CountRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,384,384->32,46,46->1
        self.conv1 = nn.Conv2d(1, 8, 3)  
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)  

        self.conv2 = nn.Conv2d(8, 16, 3)  
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)   

        self.conv3 = nn.Conv2d(16, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2) 

        self.fc1 = nn.Linear(32 * 46 * 46, 128) 
        self.fc2 = nn.Linear(128, 1)      

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        #print(x.shape)
        x = x.view(-1, 32 * 46 * 46)    
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)              
        #print(x.shape)
        return x  

def regression_model():
    return CountRegression()