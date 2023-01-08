import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input):
        #embedded = self.embedding(input).view(1, 1, -1)
        #output = embedded
        output, hidden = self.gru(input)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

