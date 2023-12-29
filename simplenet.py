import torch
import torch.nn as nn
import torch.nn.functional as F

class Simplenet(nn.Module):
    
    def __init__(self,seed=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.linear = nn.Linear(3, 3)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        

        self.apply(self._init_weights)        
#        nn.init.zeros_(self.conv1.weight) # 重みの初期値を設定
#        nn.init.ones_(self.conv1.bias)    # バイアスの初期値を設定

    def _init_weights(self,module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.1)
            module.bias.data.zero_()
            #module.bias.data.normal_(mean=0.0, std=0.001)

    def initparam(self,seed=0):
        self.apply(self._init_weights)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x