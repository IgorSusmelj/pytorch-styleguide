import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()

        self.features = nn.Sequential(
            # C x H x W
            # 3 x 32 x 32
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32, 16, 16
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64, 8, 8
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128, 4, 4
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*128, num_classes)
        )
    
    def forward(self, x):
        feat = self.features(x)
        out = self.fc(feat)
        return out