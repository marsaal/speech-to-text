import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=128, embedding_dim=32, num_classes=30):
        super().__init__()
        
        # Simple feature extraction
        self.feature_extractor = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Token embedding for previous predictions
        self.embedding = nn.Embedding(num_classes, embedding_dim, dtype=torch.float32)

        
        # Encoder RNN
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        
        # Decoder RNN
        self.decoder = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        
        # Output layer
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
        x = F.relu(self.feature_extractor(x))  # (batch, hidden_dim, time)
        x = x.transpose(1, 2)  # (batch, time, hidden_dim)

        
        # Encode the audio sequence
        encoder_outputs, encoder_hidden = self.encoder(x)
        
        # Initialize decoder with encoder's final hidden state
        decoder_hidden = encoder_hidden
        
        # Start with the first token (usually a start token)
        decoder_input = prev_tokens[:, 0].unsqueeze(1)
        
        # Store all outputs
        outputs = x.new_zeros(batch_size, max_len, self.classifier.out_features)

        
        # Generate sequence
        for t in range(max_len):
            # Get embedding of current token
            embedded = self.embedding(decoder_input)
            
            # Use a simple "context" - just take the last encoder output for each item in batch
            context = encoder_outputs[:, -1:, :]
            
            # Concatenate embedding and context as input to decoder
            decoder_input_combined = torch.cat([embedded, context], dim=2)
            
            # Run decoder for one step
            decoder_output, decoder_hidden = self.decoder(decoder_input_combined, decoder_hidden)
            # Generate prediction
            prediction = self.classifier(decoder_output.squeeze(1))
            
            # Store prediction
            outputs[:, t] = prediction
            
            # Teacher forcing: use ground truth or predicted token as next input
            use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio) and self.training
            
            if use_teacher_forcing and t < max_len - 1:
                decoder_input = prev_tokens[:, t+1].unsqueeze(1)
            else:
                # Use our prediction
                top_token = prediction.argmax(1).unsqueeze(1)  # Get most likely token
                decoder_input = top_token
        
        return outputs
