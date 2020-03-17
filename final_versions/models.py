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


class ThreeLayerLSTMnet(nn.Module):
    '''
    Create Basic LSTM:
    3 layers

    '''
    def __init__(self, input_size, hidden_size, output_dim, dropout):
        super(ThreeLayerLSTMnet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=5, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h=None):
        x = x.permute(2, 0, 1)
        if type(h) == type(None):
            out, hn = self.rnn(x)
        else:
            out, hn = self.rnn(x, h.detach())
        out = self.fc(out[-1, :, :])
        return out


class FiveLayerLSTMnet(nn.Module):
    '''
    Create Basic LSTM:
    5 layers

    '''
    def __init__(self, input_size, hidden_size, output_dim, dropout):
        super(FiveLayerLSTMnet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=5, dropout=dropout)
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
        

class GRU2HiddenDimsnet(nn.Module):
    '''
    Create Basic GRU:
    2 layers (hidden dims=hidden_dims, 2)
    
    '''
    def __init__(self, input_size, hidden_size, output_dim, dropout):
        super(GRU2HiddenDimsnet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.rnn1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, dropout=dropout)
        self.rnn2 = nn.GRU(input_size=hidden_size, hidden_size=25, num_layers=1, dropout=dropout)
        self.fc = nn.Linear(25, output_dim)
    
    def forward(self, x, h=None):
        x = x.permute(2, 0, 1)
        out, hn = self.rnn1(x)
        out, hn = self.rnn2(out)
        out = self.fc(out[-1, :, :])
        return out


class ThreeLayerGRUnet(nn.Module):
    '''
    Create Basic GRU:
    3 layers
    
    '''
    def __init__(self, input_size, hidden_size, output_dim, dropout):
        super(ThreeLayerGRUnet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=3, dropout=dropout)
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
            nn.Conv1d(cnn_input_size, rnn_input_size, kernel_size=10, stride=2),
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


class CNN2LSTM(nn.Module):
    '''
    CNN1 + LSTM1 + CNN2 + LSTM2
    
    '''
    def __init__(self, cnn1_input_size, rnn1_input_size, hidden_size1, rnn2_input_size, hidden_size2, output_dim, dropout):
        super(CNN2LSTM, self).__init__()
        self.cnn1_input_size = cnn1_input_size
        self.rnn1_input_size = rnn1_input_size
        self.rnn2_input_size = rnn2_input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_dim = output_dim
        self.cnn1 = nn.Sequential(
            nn.Conv1d(cnn1_input_size, rnn1_input_size, kernel_size=10, stride=2),
            nn.BatchNorm1d(rnn1_input_size),
            nn.ReLU(),
        )
        self.rnn1 = nn.LSTM(input_size=rnn1_input_size, hidden_size=hidden_size1, num_layers=1, dropout=dropout)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(hidden_size1, rnn2_input_size, kernel_size=10, stride=2),
            nn.BatchNorm1d(rnn2_input_size),
            nn.ReLU(),
        )
        self.rnn2 = nn.LSTM(input_size=rnn2_input_size, hidden_size=hidden_size2, num_layers=1, dropout=dropout)
        self.fc = nn.Linear(hidden_size2, output_dim)
    
    def forward(self, x, h=None):
        out = self.cnn1(x)
        out = out.permute(2,0,1)
        out, hn = self.rnn1(out)      
        out = out.permute(1, 2, 0)        
        out = self.cnn2(out)
        out = out.permute(2,0,1)
        out, hn = self.rnn2(out) 
        out = self.fc(out[-1, :, :])
        return out


class CNN2GRUnet(nn.Module):
    '''
    Conv(temporal) + Conv(spatial) + GRU
    
    '''
    def __init__(self, hidden_size, output_dim, dropout):
        super(CNN2GRUnet, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        # conv on temporal
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        # conv on spatial 
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(22, 1), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1,4)),
        )
        
        # conv on temporal
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1,4)),
        )
        # conv on temporal
    
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 10), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1,4)),
            nn.Dropout(p=dropout)
        )
    
        self.rnn1 = nn.GRU(input_size=128, hidden_size=64, num_layers=1)
        self.rnn2 = nn.GRU(input_size=64, hidden_size=32, num_layers=1)
        self.fc = nn.Linear(32, output_dim)
    
    def forward(self, x, h=None):
        x = torch.unsqueeze(x, 1)
        out = self.cnn1(x)
        out = self.cnn2(out)        
        out = self.cnn3(out)
        out = self.cnn4(out)    
        out = torch.squeeze(out)
        out = out.permute(2,0,1)

        if type(h) == type(None):
            out, hn = self.rnn1(out)
        else:
            out, hn = self.rnn1(out, h.detach())
        out, hn = self.rnn2(out)
        out = self.fc(out[-1, :, :])
        return out