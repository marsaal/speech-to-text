import torch
import torch.nn as nn
import torch.nn.functional as F

# CTC_tok_12 and CTC_tok_8
class DeepCTCModel(nn.Module):
    def __init__(self, num_classes, input_dim=80, hidden_dim=256, num_layers=3):
        super().__init__()

        # Feature extractor amb diverses convolucions
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Encoder bidireccional profund
        self.encoder = nn.GRU(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Classifier: concat de direccions → hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch, time, features)
        x = self.feature_extractor(x)  # → (batch, channels, time)
        x = x.transpose(1, 2)  # → (batch, time, channels)

        x, _ = self.encoder(x)  # → (batch, time, hidden_dim * 2)
        logits = self.classifier(x)  # → (batch, time, num_classes)
        return logits
