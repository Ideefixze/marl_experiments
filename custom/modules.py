import torch
from torch import nn


class ModuleLSTM(nn.Module):
    def __init__(self, input, output, hidden_size=128):
        super(ModuleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(input, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float)
        h_t2 = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float)
        c_t2 = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float)

        for input_t in input.split(24, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs