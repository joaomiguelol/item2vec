import torch
from torch import nn

class Item2Vec(nn.Module):
    def __init__(self, num_items, embedding_dim, input_tokens=30522):
        super().__init__()
        # embedding layer
        self.embeddings = nn.Embedding(num_items, embedding_dim)
        
        self.text_embeddings = nn.Embedding(input_tokens, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=1, batch_first=True)

        # dense layer
        self.linear = nn.Linear(embedding_dim + embedding_dim, 256)
        # activation function
        self.act = nn.ReLU()
        # dropout
        self.dropout = nn.Dropout(0.2)
        # output layer
        self.output = nn.Linear(256, 128)
        # output activation
        # self.output_act = nn.Sigmoid()

    def forward(self, item1, text):
        embed = self.embeddings(item1)
        # LSTM on text
        text = self.text_embeddings(text)

        lstm_out, _ = self.lstm(text)

        # Take the last hidden state
        text = lstm_out[:, -1, :]

        # Concatenate item embeddings and text embeddings
        combined = torch.cat((embed, text), dim=1)

        embed1 = self.dropout(combined)
        # pass through dense layer
        dense1 = self.linear(embed1)
        # pass through activation function
        act1 = self.act(dense1)
        # pass through dropout
        # pass through output layer
        output = self.output(act1)
        # pass through output activation
        # output = self.output_act(output)

        return output