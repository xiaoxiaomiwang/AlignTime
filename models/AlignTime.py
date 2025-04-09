import torch
import torch.nn as nn
from layers.Invertible import RevIN

class PECBlock(nn.Module):
    def __init__(self, input_dim, conv_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_dim, kernel_size=1, bias=False)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=conv_dim, out_channels=input_dim, kernel_size=1, bias=False)
    def forward(self, x):
        return self.conv2(self.gelu(self.conv1(x)))

class PASBlock(nn.Module):
    def __init__(self, input_dim, conv_dim, highest_freq_period):
        super(PASBlock, self).__init__()
        self.highest_freq_period = highest_freq_period
        self.sampling_sps = nn.ModuleList([
            PECBlock(input_dim // self.highest_freq_period, conv_dim) for _ in range(self.highest_freq_period)
        ])

    def FSABlock(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.highest_freq_period] = x_pad
        return y[:, :, :]

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.sampling_sps):
            x_samp.append((samp(x[:, :, idx::self.highest_freq_period].permute(0,2,1))).permute(0,2,1))
        x = self.FSABlock(x.shape, x_samp)
        return x

class SamplingBlock(nn.Module):
    def __init__(self, tokens_dim, hidden_dim, highest_freq_period):
        super().__init__()
        self.Sampling = PASBlock(tokens_dim, hidden_dim, highest_freq_period)

    def forward(self, x):
        y = x
        y = y.transpose(1, 2)
        y = self.Sampling(y)
        y = y.transpose(1, 2)
        return y

class PredictedProjection(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linears = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = self.linears(x.transpose(1, 2)).transpose(1, 2)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.highest_freq_period = configs.highest_freq_period
        self.seq_len = configs.seq_len
        if self.seq_len % self.highest_freq_period == 0:
            self.pad_len = self.seq_len
        else:
            self.pad_len = (self.seq_len // self.highest_freq_period + 1) * self.highest_freq_period
        self.sampling_block = nn.ModuleList([
            SamplingBlock(self.pad_len, configs.d_model, configs.highest_freq_period) for _ in range(configs.e_layers)
        ])
        self.norm = nn.LayerNorm(configs.enc_in)
        self.projection = PredictedProjection(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.enc_in)

    def pad_time_series(self, x):
        batch_size, seq_len, num_features = x.shape
        remainder = seq_len % self.highest_freq_period
        if remainder == 0:
            return x, seq_len
        pad_len = (seq_len//self.highest_freq_period+1)*self.highest_freq_period - seq_len
        pad_data = x[:, -pad_len:, :].flip(dims=[1])
        pad_data = torch.cat([x, pad_data], dim=1)
        return pad_data, seq_len

    def forward(self, x):
        x = self.rev(x, 'norm')
        x_padded, pad_length = self.pad_time_series(x)
        for block in self.sampling_block:
            x = block(x_padded)
        x = self.projection(x[:, :self.seq_len, :])
        x = self.rev(x, 'denorm')
        return x
