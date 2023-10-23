import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import binary_cross_entropy 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset
import pandas as pd

@torch.no_grad()
#Validation process
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)
#Training process
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] # Seguimiento del learning rate

## GPU usage

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu') #utilizar la gpu si está disponible

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    #Determina el tipo de estructura de dato, si es una lista o tupla la secciona en su subconjunto para mandar toda la información a la GPU
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl) #Mandar los data_loader que tienen todos los batch hacia la GPU


##Matplotlib plotting

def plot_losses(history):
    losses_val = [x['val_loss'] for x in history]
    losses_train = [x['train_loss'] for x in history]
    fig, ax = plt.subplots(figsize = (7,7), dpi = 100)
    ax.plot(losses_val, marker = 'x', color = 'r', label = 'Cross-Validation' )
    ax.plot(losses_train, marker = 'o', color = 'g', label = 'Training' )
    ax.set(ylabel = 'Loss', xlabel = 'Epoch', title = 'Loss vs. No. of epochs')
    plt.legend()
    plt.show()
def plot_metrics(history, metric):
    score_val = [x[metric] for x in history]
    _, ax = plt.subplots(figsize = (7,7), dpi = 100)
    ax.plot(score_val, marker = 'x', color = 'r')
    ax.set(ylabel = metric, xlabel = 'Epoch', title = f'{metric} vs. No. of epochs')
    plt.legend()
    plt.show()

def accuracy(outputs, targets):
    predictions = (outputs > 0.5).float()  # Convertir a 1 si la salida es mayor que 0.5, 0 de lo contrario
    correct = (predictions == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy

def Precision(outputs, targets):
    true_positives = (outputs * targets).sum()
    false_positives = (outputs * (1 - targets)).sum()
    precision = true_positives / (true_positives + false_positives)
    return precision

def Recall(outputs, targets):
    true_positives = (outputs * targets).sum()
    false_negatives = ((1 - outputs) * targets).sum()
    recall = true_positives / (true_positives + false_negatives)
    return recall

def F1_score(outputs, targets):
    precision_value = Precision(outputs, targets)
    recall_value = Recall(outputs, targets)
    f1 = 2 * (precision_value * recall_value) / (precision_value + recall_value)
    return f1

#classification module
class RemClassificationBase(nn.Module):
    def fit(self, epochs, lr, train_loader, val_loader,
                      weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam, lr_sched=None, start_factor = 1, end_factor = 1e-4):
        torch.cuda.empty_cache()
        history = [] # Seguimiento de entrenamiento

        # Poner el método de minimización personalizado
        optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay)
        #Learning rate scheduler
        if lr_sched is not None:    
            try:
                sched = lr_sched(optimizer, lr, epochs=epochs,steps_per_epoch=len(train_loader))
            except TypeError:
                try:
                    sched = lr_sched(optimizer, start_factor = start_factor, end_factor=end_factor, total_iters = epochs)
                except TypeError:
                    sched = lr_sched(optimizer, step_size = round(epochs/4), gamma = 0.9)
        for epoch in range(epochs):
            # Training Phase
            self.train()  #Activa calcular los vectores gradiente
            train_losses = []
            if lr_sched is not None:
                lrs = []
            for batch in train_loader:
                # Calcular el costo
                loss = self.training_step(batch)
                #Seguimiento
                train_losses.append(loss)
                #Calcular las derivadas parciales
                loss.backward()

                # Gradient clipping, para que no ocurra el exploding gradient
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)

                #Efectuar el descensod e gradiente y borrar el historial
                optimizer.step()
                optimizer.zero_grad()
                #sched step
                if lr_sched is not None:
                    lrs.append(get_lr(optimizer))
                    sched.step()

            # Fase de validación
            result = evaluate(self, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
            if lr_sched is not None:
                result['lrs'] = lrs
                self.epoch_end_one_cycle(epoch, result) #imprimir en pantalla el seguimiento
            else:
                self.epoch_end(epoch, result)

            history.append(result) # añadir a la lista el diccionario de resultados
        return history
    
    def training_step(self, batch):
        inputs, targets = batch
        # Reshape target tensor to match the input size
        out, _ = self(inputs)                  # Generar predicciones
        loss = binary_cross_entropy(out, targets) # Calcular el costo
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        out,_ = self(inputs)                    # Generar predicciones
        loss = binary_cross_entropy(out, targets)   # Calcular el costo
        acc = accuracy(out, targets) #Calcular la precisión
        score = F1_score(out, targets) 
        recall = Recall(out, targets)
        precision = Precision(out, targets)
        return {'val_loss': loss.detach(), 'val_acc': acc, 'f1_score': score, 'Recall': recall, 'Precision': precision}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()   # Sacar el valor expectado de todo el conjunto de precisión
        batch_score = [x['f1_score'] for x in outputs]
        epoch_score = torch.stack(batch_score).mean()
        batch_recall = [x['Recall'] for x in outputs]
        epoch_recall = torch.stack(batch_recall).mean()
        batch_prec = [x['Precision'] for x in outputs]
        epoch_prec = torch.stack(batch_prec).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'f1_score': epoch_score.item(), 'Recall': epoch_recall.item(), 'Precision':epoch_prec.item()}

    def epoch_end_one_cycle(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]\n\tlast_lr: {:.5f}\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\tval_acc: {:.4f}\n\tf1_score: {:.4f}\n\trecall: {:.4f}\n\tprecision: {:.4f}".format(
    epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc'], result['f1_score'], result['Recall'], result['Precision']))

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}]\n\ttrain_loss: {:.4f}\n\tval_loss: {:.4f}\n\tval_acc: {:.4f}\n\tf1_score: {:.4f}\n\trecall: {:.4f}\n\tprecision: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc'], result['f1_score'], result['Recall'], result['Precision']))
## Model module

# Deep Neural Network
def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    )
    return out
## Deep Neural Network
class DeepNeuralNetwork(RemClassificationBase):
    def __init__(self, input_size = 4, *args):
        super(DeepNeuralNetwork, self).__init__()
        
        self.overall_structure = nn.Sequential()
        #Model input and hidden layer
        for num, output in enumerate(args):
            self.overall_structure.add_module(name = f'layer_{num+1}', module = SingularLayer(input_size, output))
            input_size = output

        #Model output layer
        self.output_layer = nn.Sequential(
                nn.Linear(input_size, 1),
                nn.Sigmoid()
            )
    def forward(self, xb):
        out = self.overall_structure(xb)
        out = self.output_layer(out)
        return out

#RNNs
##Bidirectional RNN
class BidirectionalRNN(RemClassificationBase):
    def __init__(self, rnn1, rnn2, architecture):
        super(BidirectionalRNN, self).__init__()
        self.rnn1 = rnn1
        self.rnn2 = rnn2
        self.fc = nn.Sequential(
            DeepNeuralNetwork(rnn1.hidden_size+rnn2.hidden_size, 1, *architecture),
            nn.Sigmoid()
)
    def forward(self, x):
        # Forward pass through the first RNN
        _, hidden1 = self.rnn1(x)
        # Reverse the input sequence for the second RNN
        x_backward = torch.flip(x, [1])
        
        # Forward pass through the second RNN
        _, hidden2 = self.rnn2(x_backward)

        # Concatenate to bidirectional output
        hidden_bidirectional = torch.cat((hidden1,hidden2), dim = 1)
        
        out = self.fc(hidden_bidirectional)

        return out, hidden_bidirectional
##Deep LSTM
class DefaultLSTM(RemClassificationBase):
    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.2, bidirectional=True):
        super(DefaultLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional=bidirectional, dropout = dropout)
        # Output layer
        if bidirectional:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size*2,1),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size,1),
                nn.Sigmoid()
            )
            
    def forward(self, x):
        lstm_out, (hn,_) = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Take the output from the last time step
        return output, hn    
class DefaultGRU(RemClassificationBase):
    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.2, bidirectional=True):
        super(DefaultGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        # LSTM layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True, bidirectional=bidirectional, dropout = dropout)
        # Output layer
        if bidirectional:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size*2,1),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size,1),
                nn.Sigmoid()
            )
            
    def forward(self, x):
        gru_out, hn = self.gru(x)
        output = self.fc(gru_out[:, -1, :])  # Take the output from the last time step
        return output, hn    

class DeepLSTM(RemClassificationBase):
    def __init__(self, hidden_size, input_size, mlp_architecture):
        super(DeepLSTM, self).__init__()
        self.hidden_size = hidden_size
        #Forget gate
        self.F_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.F_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Input gate
        self.I_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.I_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Ouput gate
        self.O_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.O_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Input node
        self.C_hat_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.C_hat_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)  
        ### to output
        self.fc = nn.Sequential(
            DeepNeuralNetwork(hidden_size, 1, *mlp_architecture),
            nn.Sigmoid()
        )  
    def forward(self,x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        cn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)

        for t in range(sequence_size):
            xt = x[:, t, :]
            #forward
            a_F = self.F_h(hn) + self.F_x(xt)
            F = torch.sigmoid(a_F) #forget gate
            a_I = self.I_h(hn) + self.I_x(xt)
            I = torch.sigmoid(a_I) #input gate
            a_O = self.O_h(hn) + self.O_x(xt)
            O = torch.sigmoid(a_O) #output gate
            a_C_hat = self.C_hat_h(hn) + self.C_hat_x(xt)
            C_hat = torch.tanh(a_C_hat)
            cn = F*cn + I*C_hat
            hn = O*torch.tanh(cn)
        out = self.fc(hn)
        return out, hn

class DeepGRU(RemClassificationBase):
    def __init__(self, hidden_size, input_size, mlp_architecture):
        super(DeepGRU, self).__init__()
        self.hidden_size = hidden_size
        #Update gate
        self.Z_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.Z_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Reset gate
        self.R_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.R_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        #Possible hidden state
        self.H_hat_h = DeepNeuralNetwork(hidden_size, hidden_size, *mlp_architecture)
        self.H_hat_x = DeepNeuralNetwork(input_size, hidden_size, *mlp_architecture)
        ### to output
        self.fc = nn.Sequential(
            DeepNeuralNetwork(hidden_size, 1, *mlp_architecture),
            nn.Sigmoid()
        )  
    def forward(self, x):
        batch_size, sequence_size, _ = x.size()
        hn = torch.zeros(batch_size, self.hidden_size, requires_grad = True)
        for t in range(sequence_size):
            xt = x[:, t, :]
            Z = torch.sigmoid(self.Z_h(hn)+self.Z_x(xt))
            R = torch.sigmoid(self.R_h(hn)+self.R_x(xt))
            H_hat = torch.tanh(self.H_hat_h(hn*R)+self.H_hat_x(xt))
            hn = hn*Z + (torch.ones_like(Z)-Z)*H_hat
        out = self.fc(hn)
        return out, hn

## DATA and DATASET

def get_individual():
    id_ = [  46343,  759667,  781756,  844359, 1066528, 1360686, 1449548,
       1455390, 1818471, 2598705, 2638030, 3509524, 3997827, 4018081,
       4314139, 4426783, 5132496, 5383425, 5498603, 5797046, 6220552,
       7749105, 8000685, 8173033, 8258170, 8530312, 8686948, 8692923,
       9106476, 9618981, 9961348]
    data_list = []
    
    df = pd.read_csv('data/data.csv')
    
    for i in id_:
        data = df.loc[df['Unnamed: 0'] == i, :].drop(['Unnamed: 0', 'Unnamed: 1', 'cosine', 'time'], axis =1)
        data_list.append(data)
    
    return data_list

class SequentialDataset(Dataset):
    def __init__(self, inputs, targets, sequence_length):
        self.x_scaler = StandardScaler()
        inputs = self.x_scaler.fit_transform(inputs)
        self.inputs = torch.from_numpy(inputs).to(torch.float32)
        self.targets = torch.from_numpy(targets).to(torch.float32)
        self.sequence_length= sequence_length
    def __len__(self):
        return len(self.inputs) - self.sequence_length
    def __getitem__(self, idx):
        input_sequence = self.inputs[idx:idx + self.sequence_length, :]
        target = self.targets[idx + self.sequence_length-1]
        return input_sequence, target

class MegaDataset(Dataset):
    def __init__(self, data_list, sequence_length):
        self.datasets = [SequentialDataset(dataset.values[:, :-1], dataset.values[:, -1].reshape(-1,1), sequence_length) for dataset in data_list]
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        # Determine which dataset the sample belongs to
        dataset_idx = 0
        cumulative_length = len(self.datasets[0])
        while index >= cumulative_length:
            dataset_idx += 1
            cumulative_length += len(self.datasets[dataset_idx])

        # Adjust the index relative to the chosen dataset
        if dataset_idx > 0:
            index -= sum(len(self.datasets[i]) for i in range(dataset_idx))

        return self.datasets[dataset_idx][index]