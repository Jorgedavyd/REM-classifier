from utils import *
from data.data_utils import *
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import cross_entropy
from torch.utils.data import random_split, TensorDataset
#Hyperparameters

##individual_classifier
epochs = 2000
opt = torch.optim.Adam
lr = 0.001
criterion = binary_cross_entropy
batch_size = 128
hidden_size = 51  
num_layers = 10 #Number of hidden layers of the internal LSTM
input_size = 3 #Size of vector

##metaclassifier

epochs_meta = 1000
opt_meta = torch.optim.Adam
lr_meta = 0.001
criterion_meta = binary_cross_entropy

#data


dataframe_list = get_individual()

data_loaders = []

device = get_default_device()


for dataframe in dataframe_list:
    output_cols = dataframe.columns.values[-1]
    input_cols = dataframe.columns.values[:-1]
    
    #Sending the data to torch
    inputs, targets = dataframe_to_torch(dataframe,input_cols, output_cols)

    #Creating the dataset
    dataset = TensorDataset(inputs, targets)

    #Generating 
    val_size = round(0.2*len(dataset))
    train_ds, val_ds = random_split(dataset , [len(dataset) - val_size, val_size])

    #Defining dataloader
    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, pin_memory=True, num_workers=4, shuffle=True)

    #to gpu if available
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    #append dataloader to the individual list
    data_loaders.append((train_loader, val_loader))

#

#training individual models
models = []

for (train_loader, val_loader) in data_loaders:
    #Defining model
    model = to_device(LSTMModel(input_size, hidden_size, num_layers), device)
    #Defining optimizer
    opt = opt(model.parameters(), lr=lr)
    #Training and validation step
    for epoch in tqdm(range(epochs)):
        model.train()
        for features, targets in train_loader:
            pred = model(features)
            train_loss = criterion(pred, targets)
            train_loss.backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        for features, targets in val_loader:
            pred = model(features)
            val_loss = criterion(pred, targets)
            val_acc = accuracy(pred, targets)
            score = F1_score(pred, targets)
            print(f'===============================\nEPOCH: {epoch}\nval_acc: {val_acc} \nf1_score: {score} \nval_loss: {val_loss}\ntrain_loss: {train_loss}\n')
    for param in model.parameters():
        param.requires_grad = False
        
    models.append(model)


meta_classifier_1 = RandomNeuronalPopulation(models)
meta_classifier_2 = MetaClassifierNN(models)





#Create training loop for meta classifier NN

for (train_loader, val_loader) in data_loaders:
    for epoch in tqdm(range(epochs_meta)):
        opt = opt(meta_classifier_2.parameters(), lr=lr)
        #Training and validation step
        for epoch in tqdm(range(epochs)):
            meta_classifier_2.train()
            for features, targets in train_loader:
                pred = meta_classifier_2(features)
                train_loss = criterion_meta(pred, targets)
                train_loss.backward()
                opt.step()
                opt.zero_grad()
            meta_classifier_2.eval()
            for features, targets in val_loader:
                pred = meta_classifier_2(features)
                val_loss = criterion_meta(pred, targets)
                val_acc = accuracy(pred, targets)
                score = F1_score(pred, targets)
                print(f'===============================\nEPOCH: {epoch}\nval_acc: {val_acc} \nf1_score: {score} \nval_loss: {val_loss}\ntrain_loss: {train_loss}\n')




torch.save(meta_classifier_1.state_dict(), 'LSTM_RandomNeuronalPopulation.pth')
torch.save(meta_classifier_2.state_dict(), 'LSTM_MetaClassifierNN.pth')