import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),     
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(4096, z_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x



class Classifier(nn.Module):
    # TODO: fill in this class with the required architecture and
    # TODO: associated forward method
    # raise NotImplementedError
    def __init__(self,z_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
       

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x= self.fc2(x)
        return F.log_softmax(x, dim=1)