import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels)
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.residual(x))

class DeepCTCModelV2(nn.Module):
    def __init__(self, num_classes=85, input_dim=80, hidden_dim=256, num_layers=2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ResidualBlock(input_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.encoder = nn.ModuleList([
            nn.GRU(input_size=256, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True),
            nn.GRU(input_size=hidden_dim*2, hidden_size=hidden_dim, num_layers=num_layers,batch_first=True, bidirectional=True),
            nn.GRU(input_size=hidden_dim*2, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True),
        ])


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # (batch, channels, time)
        x = x.transpose(1, 2)          # (batch, time, channels)
        for gru in self.encoder:
            x, _ = gru(x)              # (batch, time, hidden_dim * 2)     
        logits = self.classifier(x)    # (batch, time, num_classes)
        return logits

