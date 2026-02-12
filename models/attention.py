import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        hidden = hidden[-1].unsqueeze(1).repeat(1, timestep, 1)  # (batch, seq_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, hidden_dim)
        energy = energy @ self.v  # (batch, seq_len)
        return F.softmax(energy, dim=1)  # (batch, seq_len)

class AttentionModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, embedding_dim=64, num_classes=30):
        super().__init__()

        # Better feature extractor: two convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.embedding = nn.Embedding(num_classes, embedding_dim, dtype=torch.float32)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)

        self.attention = Attention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, prev_tokens, teacher_forcing_ratio=0.5):
        """
        Args:
            x: Input audio features (batch, time, feature)
            prev_tokens: Target tokens for teacher forcing (batch, sequence_len)
            teacher_forcing_ratio: Probability of using teacher forcing
        """
        batch_size = x.size(0)
        max_len = prev_tokens.size(1)
        
        # Process audio features
        # Transpose to match Conv1d expected input: (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.feature_extractor(x)  # (batch, hidden_dim, time)
        x = x.transpose(1, 2)  # (batch, time, hidden_dim)

        
        # Encode the audio sequence
        encoder_outputs, encoder_hidden = self.encoder(x)
        
        # Initialize decoder with encoder's final hidden state
        decoder_hidden = encoder_hidden
        
        # Start with the first token (usually a start token)
        decoder_input = prev_tokens[:, 0].unsqueeze(1)
        
        # Store all outputs
        outputs = x.new_zeros(batch_size, max_len, self.classifier.out_features)
        

        for t in range(max_len):
            # Get embedding of current token
            embedded = self.embedding(decoder_input)  

            # Attention over encoder outputs
            attn_weights = self.attention(decoder_hidden, encoder_outputs)  
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  

            decoder_input_combined = torch.cat([embedded, context], dim=2)  

            decoder_output, decoder_hidden = self.decoder(decoder_input_combined, decoder_hidden)
            prediction = self.classifier(decoder_output.squeeze(1))  

            outputs[:, t] = prediction

            use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio) and self.training
            if use_teacher_forcing and t < max_len - 1:
                decoder_input = prev_tokens[:, t + 1].unsqueeze(1)
            else:
                decoder_input = prediction.argmax(1).unsqueeze(1)

        return outputs