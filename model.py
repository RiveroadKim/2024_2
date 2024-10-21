import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class AcousticSceneClassifier(nn.Module):
    def __init__(self, num_classes=15, dropout_rate=0.2):
        super(AcousticSceneClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.attention1 = SelfAttention(64)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = self._make_layer(64, 128, 2)
        self.attention2 = SelfAttention(128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = self._make_layer(128, 256, 2)
        self.attention3 = SelfAttention(256)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.attention1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.attention2(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.attention3(x)
        x = self.dropout3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout4(x)
        x = self.fc(x)

        return x

# 모델 사용 예시
if __name__ == "__main__":
    model = AcousticSceneClassifier()
    input_tensor = torch.randn(32, 1, 40, 501)  # [B, C, H, W]
    output = model(input_tensor)
    print("출력 텐서 크기:", output.shape)
