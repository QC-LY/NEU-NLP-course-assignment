import torch
from torch import sigmoid, tanh, cat
import torch.nn as nn
from torch.nn import Parameter
import math

class MyLSTM(nn.Module):
    """
    LSTM
    """
    def __init__(self, inputs_size, hidden_size):
        super().__init__()
        self.inputs_size = inputs_size
        self.hidden_size = hidden_size
        self.W = Parameter(torch.Tensor(inputs_size, hidden_size * 4))
        self.U = Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def forward(self, x, init_states=None):
        """
         x -> (batch, sequence, feature)
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (sigmoid(gates[:, :HS]), sigmoid(gates[:, HS:HS * 2]), tanh(gates[:, HS * 2:HS * 3]), sigmoid(gates[:, HS * 3:]))
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

    def init_weights(self):

        STDV = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-STDV, STDV)