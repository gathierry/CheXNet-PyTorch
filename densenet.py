from torch import nn
import torchvision
import torch.nn.functional as F


class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.features = densenet121.features
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        features = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        out = self.classifier(out)
        return out, features


if __name__ == '__main__':
    import torch
    net = DenseNet121()
    out = net(torch.randn(1, 3, 512, 512))
    print(out.size())