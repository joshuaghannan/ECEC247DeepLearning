import numpy as np
import torch
import torch.nn as nn

'''
Add your new models here as a new class

'''

class LSTMnet(nn.Module):
    '''
    Create Basic LSTM:
    2 layers

    '''
    def __init__(self, input_size, hidden_size, output_dim, dropout):
        super(LSTMnet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h=None):
        x = x.permute(2, 0, 1)
        if type(h) == type(None):
            out, hn = self.rnn(x)
        else:
            out, hn = self.rnn(x, h.detach())
        out = self.fc(out[-1, :, :])
        return out


class CNNLSTMnet(nn.Module):
    '''
    CNN + LSTM
    
    '''
    def __init__(self, cnn_input_size, rnn_input_size, hidden_size, output_dim, dropout):
        super(CNNLSTMnet, self).__init__()
        self.cnn_input_size = cnn_input_size
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(cnn_input_size, rnn_input_size, kernel_size=10, stride=2),
            nn.BatchNorm1d(rnn_input_size),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h=None):
        out = self.cnn(x)
        out = out.permute(2,0,1)
        if type(h) == type(None):
            out, hn = self.rnn(out)
        else:
            out, hn = self.rnn(out, h.detach())
        out = self.fc(out[-1, :, :])
        return out

class CNN2LSTM(nn.Module):
    '''
    CNN1 + LSTM1 + CNN2 + LSTM2
    
    '''
    def __init__(self, cnn1_input_size, rnn1_input_size, hidden_size1, cnn2_input_size, rnn2_input_size, hidden_size2, output_dim, dropout):
        super(CNNLSTMnet, self).__init__()
        self.cnn_input_size = cnn_input_size
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.cnn1 = nn.Sequential(
            nn.Conv1d(cnn1_input_size, rnn1_input_size, kernel_size=10, stride=2),
            nn.BatchNorm1d(rnn_input_size),
            nn.ReLU(),
        )
        self.rnn1 = nn.LSTM(input_size=rnn1_input_size, hidden_size=hidden_size1, num_layers=1, dropout=dropout)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(cnn2_input_size, rnn2_input_size, kernel_size=10, stride=2),
            nn.BatchNorm1d(rnn2_input_size),
            nn.ReLU(),
        )
        self.rnn2 = nn.LSTM(input_size=rnn2_input_size, hidden_size=hidden_size2, num_layers=1, dropout=dropout)
        self.fc = nn.Linear(hidden_size2, output_dim)
    
    def forward(self, x, h=None):
        out = self.cnn1(x)
        # out (batch, feature_num, seq_len)
        out = out.permute(2,0,1)
        # out (seq_len, batch, feature_num)
        if type(h) == type(None):
            out, hn = self.rnn(out)
        else:
            out, hn = self.rnn(out, h.detach())        
        # out (seq_len, batch, feature_num)
        out = out.permute(1, 2, 0)
        # out (batch, feature_num, seq_len)


        out = self.fc(out[-1, :, :])
        return out

class GRUnet(nn.Module):
    '''
    Create Basic GRU:
    2 layers
    
    '''
    def __init__(self, input_size, hidden_size, output_dim, dropout):
        super(GRUnet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h=None):
        x = x.permute(2, 0, 1)
        if type(h) == type(None):
            out, hn = self.rnn(x)
        else:
            out, hn = self.rnn(x, h.detach())
        out = self.fc(out[-1, :, :])
        return out

class CNNGRUnet(nn.Module):
    '''
    CNN + GRU
    
    '''
    def __init__(self, cnn_input_size, rnn_input_size, hidden_size, output_dim, dropout):
        super(CNNGRUnet, self).__init__()
        self.cnn_input_size = cnn_input_size
        self.rnn_input_size = rnn_input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(cnn_input_size, rnn_input_size, kernel_size=40, stride=2),
            nn.BatchNorm1d(rnn_input_size),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h=None):
        out = self.cnn(x)
        out = out.permute(2,0,1)
        if type(h) == type(None):
            out, hn = self.rnn(out)
        else:
            out, hn = self.rnn(out, h.detach())
        out = self.fc(out[-1, :, :])
        return out