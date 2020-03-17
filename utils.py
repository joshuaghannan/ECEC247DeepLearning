import numpy as np
import torch
import time
from models import *
from data_utils import *

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    # device = torch.device("cuda:1") # For Yiming 
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


def InitRNN(rnn_type="LSTM", input_size=22, rnn_input_size=40, hidden_size=50, rnn2_input_size=25, hidden_size2=20, output_dim=4, dropout=0.5, lr=1e-3, weight_decay=1e-4):
    '''
    Function to initialize RNN
    
    input: RNN type(LSTM, GRU, CNNLSTM), and other params if neccessary (regularization, acitvation, dropout, num layers, etc.)

    output: model, criterion, optimizer

    TODO: Eventually should also take in params such as dropout, number of layers, and activation function(s), etc.
    '''

    print("RNN TYPE: {}".format(rnn_type))
    print("WEIGHT DECAY: {}".format(weight_decay))
    print("LEARNING RATE: {}".format(lr))

    if rnn_type=="LSTM":
        model = LSTMnet(input_size=input_size, hidden_size=hidden_size, output_dim=output_dim, dropout=dropout).to(device)

    elif rnn_type=="ThreeLayerLSTMnet":
        model = ThreeLayerLSTMnet(input_size=input_size, hidden_size=hidden_size, output_dim=output_dim, dropout=dropout).to(device)

    elif rnn_type=="FiveLayerLSTMnet":
        model = FiveLayerLSTMnet(input_size=input_size, hidden_size=hidden_size, output_dim=output_dim, dropout=dropout).to(device)

    elif rnn_type=="GRU":
        model = GRUnet(input_size=input_size, hidden_size=hidden_size, output_dim=output_dim, dropout=dropout).to(device)
  
    elif rnn_type=="GRU2HiddenDimsnet":
        model = GRU2HiddenDimsnet(input_size=input_size, hidden_size=hidden_size, output_dim=output_dim, dropout=dropout).to(device)
  
    elif rnn_type=="ThreeLayerGRUnet":
        model = ThreeLayerGRUnet(input_size=input_size, hidden_size=hidden_size, output_dim=output_dim, dropout=dropout).to(device)
    
    elif rnn_type=="CNNLSTM":
        model = CNNLSTMnet(cnn_input_size=input_size, rnn_input_size=rnn_input_size, hidden_size=hidden_size, output_dim=output_dim, dropout=dropout).to(device)

    elif rnn_type=="CNNGRUnet":
        model = CNNGRUnet(cnn_input_size=input_size, rnn_input_size=rnn_input_size, hidden_size=hidden_size, output_dim=output_dim, dropout=dropout).to(device)

    elif rnn_type=="CNN2LSTM":
        model = CNN2LSTM(cnn1_input_size=input_size, rnn1_input_size=rnn_input_size, hidden_size1=hidden_size, rnn2_input_size=rnn2_input_size, hidden_size2=hidden_size2, output_dim=output_dim, dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, criterion, optimizer


def TrainValRNN(model, criterion, optimizer, trainloader, valloader=None, num_epochs=15, verbose=True, aug_type=None, window_size=None, vote_num=None):
    val_acc_list = []
    best_val_acc = 0.0
    for ep in range(num_epochs):
        tstart = time.time()
        running_loss = 0.0
        correct, total = 0, 0
        for idx, batch in enumerate(trainloader):
            optimizer.zero_grad()
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            output = model(X)
            loss = criterion(output, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == y).item()
            total += y.shape[0]
        train_acc = correct / total
        train_loss = running_loss
        '''
        The validation need to be customized according to the data augmenation type
        for stft and cwt: they didn't increase the number of trials, we can directly pass the augmented data to the model
        for window: it increase the number of trials, we need to do a voting for different subsequences in one trial
        
        '''
        if aug_type == 'window':
            correct, total = 0, 0
            for idx, batch in enumerate(valloader):
                #X = batch['X'].permute(2, 0, 1).to(device)
                X = batch['X'].to(device)
                y = batch['y'].to(device)
                vote_idx = np.random.choice(1000-window_size, vote_num)
                vote_pred = np.zeros(y.shape[0])
                for i in range(len(vote_idx)):
                    X_sub = X[:,:,vote_idx[i]:vote_idx[i]+window_size]
                    output = model(X_sub)
                    pred = torch.argmax(output, dim=1)
                    if i == 0:
                        vote_matrix = np.asarray(pred.cpu().view(-1, 1))
                    else:
                        vote_matrix = np.hstack((vote_matrix, np.asarray(pred.cpu().view(-1,1))))
                for row in range(y.shape[0]):
                    vote_pred[row] = np.bincount(vote_matrix[row, :]).argmax()
                vote_pred = torch.from_numpy(vote_pred).long()
                correct += torch.sum(vote_pred == y.cpu()).item()
                total += y.shape[0]
            val_acc = correct / total        
        else:
            correct, total = 0, 0
            for idx, batch in enumerate(valloader):
                X = batch['X'].to(device)
                y = batch['y'].to(device)
                output = model(X)                    
                pred = torch.argmax(output, dim=1)
                correct += torch.sum(pred == y.cpu()).item()
                total += y.shape[0]
            val_acc = correct / total
        tend = time.time()
        if verbose:
            print('epoch: {:<3d}    time: {:<3.2f}    loss: {:<3.3f}    train acc: {:<1.3f}    val acc: {:<1.3f}'.format(ep+1, tend - tstart, train_loss, train_acc, val_acc))
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model = model
            print ('saving best model...')
    return best_model

def TestRNN(model, X_test, y_test, p_test, aug_type=None, window_size=None, vote_num=None):
    if aug_type == 'window':
        EEG_testset = EEG_Dataset(X_test=X_test, y_test=y_test, p_test=p_test, mode='test')
        EEG_testloader = DataLoader(EEG_testset, batch_size=128, shuffle=False)
        correct, total = 0, 0
        for idx, batch in enumerate(EEG_testloader):
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            vote_idx = np.random.choice(1000-window_size, vote_num)
            vote_pred = np.zeros(y.shape[0])
            for i in range(len(vote_idx)):
                X_sub = X[:,:,vote_idx[i]:vote_idx[i]+window_size]
                output = model(X_sub)
                pred = torch.argmax(output, dim=1)
                if i == 0:
                    vote_matrix = np.asarray(pred.cpu().view(-1, 1))
                else:
                    vote_matrix = np.hstack((vote_matrix, np.asarray(pred.cpu().view(-1,1))))
                for row in range(y.shape[0]):
                    vote_pred[row] = np.bincount(vote_matrix[row, :]).argmax()
            vote_pred = torch.from_numpy(vote_pred).long()
            correct += torch.sum(vote_pred == y.cpu()).item()
            total += y.shape[0]
        test_acc = correct / total 
    else:
        X_test, y_test, p_test = Aug_Data(X_test, y_test, p_test, aug_type=aug_type)
        EEG_testset = EEG_Dataset(X_test=X_test, y_test=y_test, p_test=p_test, mode='test')
        EEG_testloader = DataLoader(EEG_testset, batch_size=128, shuffle=False)
        for idx, batch in enumerate(EEG_testloader):
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            output = model(X)                    
            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == y.cpu()).item()
            total += y.shape[0]
        test_acc = correct / total
    print ('Testing Accuracy: {:.4f}'.format(test_acc))