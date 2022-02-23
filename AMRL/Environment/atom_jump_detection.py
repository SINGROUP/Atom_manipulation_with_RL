import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import confusion_matrix

def cal_accuracy(y_true, y_pred):
    """
    Parameters
    ----------
    y_true, y_pred: array_like
        
    Return
    ------
    float
        accuracy of y_pred
    """
    return (y_true == y_pred).sum()/ (y_true.shape[0])

class conv_dataset(Dataset):
    def __init__(self, currents, atom_move_by, move_threshold, length=2048):
        self.currents = currents
        self.atom_move_by = atom_move_by

        self.length = length
        self.move_threshold = move_threshold
        
    def __len__(self):
        return len(self.currents)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        currents_same_len, atom_moved = [], []
        current = self.currents[idx]
        
        while len(current)<self.length:
            current = np.hstack((current,current))
        new_current = current[:self.length]
        new_current = (new_current - np.mean(new_current))/np.std(new_current)
        currents_same_len.append(new_current)
        atom_moved.append(self.atom_move_by[idx]>self.move_threshold)
        sample = {'current': np.vstack(currents_same_len), 'atom_moved': np.array(atom_moved)}
        return sample

class CONV(nn.Module):
    def __init__(self,input_dim, kernel_size, max_pool_kernel_size, stride, max_pool_stride, output_dim):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size = kernel_size, stride=stride)
        lout1 = self.get_size(input_dim, kernel_size, stride=stride)
        self.max_pool1 = nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride)
        lout1_1 = self.get_size(lout1, max_pool_kernel_size, stride=max_pool_stride)
        self.conv2 = nn.Conv1d(1, 1, kernel_size = kernel_size, stride=stride)
        lout2 = self.get_size(lout1_1, kernel_size, stride=stride)
        self.max_pool2 = nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride)
        lout2_1 = int(self.get_size(lout2, max_pool_kernel_size, stride=max_pool_stride))
        self.fc3 = nn.Linear(lout2_1, output_dim)
        self.dropout= nn.Dropout(0.1)
        self.float()
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = self.dropout(x)
        x= torch.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
    def get_size(self, Lin, kernel_size, stride = 1, padding = 0, dilation = 1):
        Lout = (Lin + 2*padding - dilation*(kernel_size-1)-1)/stride + 1
        return Lout

class AtomJumpDetector_conv:
    def __init__(self, data_len, load_weight = None, batch_size=64, move_threshold=0.1):
        self.data_len = data_len
        self.batch_size=batch_size
        self.move_threshold = move_threshold
        self.conv = CONV(data_len, 64, 4, 4, 2, 1)
        
        if load_weight is not None:
            print('Load cnn weight')
            self.load_weight(load_weight)

        self.optim = Adam(self.conv.parameters(),lr=1e-3)
        self.criterion = nn.BCELoss()
        self.currents, self.atom_move_by = [], []
        self.currents_val, self.atom_move_by_val = [], []
        
    def push(self, current, atom_move_by):
        if current is not None:
            self.currents.append(current)
            self.currents_val.append(current)
            self.atom_move_by.append(atom_move_by)
            self.atom_move_by_val.append(atom_move_by)
    
    def train(self):
        print('Training convnet')
        dset = conv_dataset(self.currents, self.atom_move_by, self.move_threshold)

        dataloader = DataLoader(dset, batch_size=self.batch_size,
                        shuffle=True, num_workers=0)
        for _, sample_batched in enumerate(dataloader):
            current = sample_batched['current']
            am = sample_batched['atom_moved']
            prediction = self.conv(current.float())
            loss = self.criterion(torch.squeeze(prediction,-1), am.type(torch.float32))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        self.currents_val, self.atom_move_by_val = [], []

    def eval(self):
        dset = conv_dataset(self.currents_val, self.atom_move_by_val, self.move_threshold)
        dataloader = DataLoader(dset, batch_size=len(dset),
                        shuffle=True, num_workers=0)

        for _, sample_batched in enumerate(dataloader):
            current = sample_batched['current']
            am = torch.squeeze(sample_batched['atom_moved']).numpy()
            prediction = torch.squeeze(self.conv(current.float())).detach().numpy()>0.5
        accuracy = cal_accuracy(am, prediction)
        cm = confusion_matrix(am, prediction, normalize='pred')
        print('Validation over {}  data. Accuracy: {}, True positive: {}, True negative: {}'.format(len(dset), accuracy, cm[1,1], cm[0,0]))
        return accuracy, cm[1,1], cm[0,0]

    def predict(self, current):
        dset = conv_dataset([current], [True], self.move_threshold)
        dataloader = DataLoader(dset, batch_size=len(dset),
                        shuffle=True, num_workers=0)
        for _, sample_batched in enumerate(dataloader):
            current = sample_batched['current']
            prediction = torch.squeeze(self.conv(current.float())).detach().numpy()
            print('Prediction:', prediction)
        return prediction>0.5, prediction
    
    def load_weight(self, load_weight):
        self.conv.load_state_dict(torch.load(load_weight))

        
        

