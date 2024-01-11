import torch
from torch import nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=9999):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)
        
        # Create the positional encodings as a constant
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Item2Vec(nn.Module):
    def __init__(self, num_items, embedding_dim, input_tokens=30522,max_len=32):
        super().__init__()
        # Embedding layer for item IDs
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        # Embedding layer for text input
        self.text_embeddings = nn.Embedding(input_tokens, embedding_dim)
        
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8),
            num_layers=3
        )

        # Dense layer
        self.linear = nn.Linear(embedding_dim + embedding_dim, 512)

        # Activation function
        self.act = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(0.2)

        # Output layer
        self.output = nn.Linear(512, embedding_dim)  # Output embedding dimension

    def forward(self, item1, text):
        # Embed item IDs and text input
        item_embed = self.item_embeddings(item1)
        text_embed = self.text_embeddings(text)
        
        text_embed = self.positional_encoding(text_embed)
        
        # Apply Transformer encoder to text embeddings
        text_encoded = self.transformer_encoder(text_embed)

        # Apply mean-pooling to obtain a single embedding for the entire sentence
        text_encoded_pooled = text_encoded.mean(dim=1)

        # Concatenate item embeddings and pooled text embeddings
        combined = torch.cat((item_embed, text_encoded_pooled), dim=1)

        # Apply dropout
        embed1 = self.dropout(combined)

        # Pass through dense layer
        dense1 = self.linear(embed1)

        # Pass through activation function
        act1 = self.act(dense1)

        # Pass through the output layer to get the final aggregated embedding
        output = self.output(act1)

        return output
