import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=23, d_model=768, nhead=12, num_layers=12, num_classes=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Enhanced character-level CNN with residual connections
        self.char_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model if i == 0 else d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(0.2),
            ) for i in range(3)
        ])
        
        # Multi-head attention with relative positional encoding
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.2,
            batch_first=True
        )
        
        # Transformer encoder with pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced classification head with class balancing
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Class weights for balancing
        self.class_weights = torch.ones(num_classes)
        self.class_weights[0] = 0.1  # Reduce weight for class 0
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, attention_mask=None):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # CNN processing with residual connections
        x_conv = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        for conv_layer in self.char_conv:
            x_conv_new = conv_layer(x_conv)
            x_conv = x_conv + x_conv_new  # Residual connection
        x = x_conv.transpose(1, 2)  # (batch_size, seq_len, d_model)
        
        # Self-attention with relative positional encoding
        if attention_mask is not None:
            attn_output, _ = self.self_attn(x, x, x, key_padding_mask=~attention_mask)
        else:
            attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output  # Residual connection
        
        # Transformer processing
        if attention_mask is not None:
            x = self.transformer(x, src_key_padding_mask=~attention_mask)
        else:
            x = self.transformer(x)
        
        # Classification with class weights
        x = self.classifier(x)  # (batch_size, seq_len, num_classes)
        
        # Apply class weights
        x = x * self.class_weights.to(x.device)
        
        return x