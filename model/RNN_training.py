from utils import *
from data.data_utils import *
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, TensorDataset
#Hyperparameters

##individual_classifier
epochs = 10
opt_ind = torch.optim.Adam
lr = 7e-4
criterion = binary_cross_entropy
hidden_size = 100  
num_layers = 10   #Number of hidden layers of the internal LSTM
input_size = 2   #Size of vector

##metaclassifier
epochs_meta = 100
opt_meta = torch.optim.Adam
lr_meta = 1e-4
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

    
    sequence_length = round(len(inputs)/5)

    #Dataset
    dataset = SequentialDataset(inputs, targets, sequence_length)

    #Generating 
    val_size = round(0.2*len(dataset))
    train_ds, val_ds = random_split(dataset , [len(dataset) - val_size, val_size])

    
    batch_size = round(len(train_ds)/10)


    #Defining dataloader
    train_loader = DataLoader(train_ds, batch_size=batch_size, drop_last = True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, drop_last = True, shuffle=True)

    #to gpu if available
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    #append dataloader to the individual list
    data_loaders.append((train_loader, val_loader))



#training individual models
models = []
for (train_loader, val_loader) in data_loaders:
    epochs=50
    #Defining model
    model = to_device(LSTMModel(input_size, hidden_size, num_layers), device)
    while True:
        #Defining optimizer
        opt = opt_ind(model.parameters(), lr=lr)
        #Training and validation step
        for epoch in tqdm(range(epochs)):
            model.train()
            for batch in train_loader:
                x, y = batch
                yhat = model(x)
                J_train = criterion(yhat, y.unsqueeze(-1))
                J_train.backward()
                opt.step()
                opt.zero_grad()
            model.eval()
            for batch in val_loader:
                x, y = batch
                yhat = model(x)
                J_val = criterion(yhat, y.unsqueeze(-1))
                val_acc = accuracy(yhat, y.unsqueeze(-1))
                score = F1_score(yhat, y.unsqueeze(-1))
                print(f'===============================\nEPOCH: {epoch}\nval_acc: {val_acc} \nf1_score: {score} \nval_loss: {J_val}\ntrain_loss: {J_train}\n')
        ask = input('1. Change lr.\n2. Train with these hyperparameters for 25 epochs. \n3. Next model.\n4. Restart model.\n')
        if ask=='1':
            epochs = int(input('epochs: '))
            lr=float(input('lr: '))
            continue
        elif ask=='2':
            epochs = 25
            continue
        elif ask=='3':
            for param in model.parameters():
                param.requires_grad = False
            models.append(model)
            break
        elif ask=='4':
            epochs = int(input('epochs: '))
            lr=float(input('lr: '))
            model = to_device(LSTMModel(input_size, hidden_size, num_layers), device)

meta_classifier_1 = RandomNeuronalPopulation(models)
meta_classifier_2 = MetaClassifierNN(models)


#Create training loop for meta classifier NN

for (train_loader, val_loader) in data_loaders:
    for epoch in tqdm(range(epochs_meta)):
        opt = opt_meta(meta_classifier_2.parameters(), lr=lr)
        #Training and validation step
        for epoch in tqdm(range(epochs)):
            meta_classifier_2.train()
            for batch in train_loader:
                x, y = batch
                yhat = model(x)
                J_train = criterion(yhat, y.unsqueeze(-1))
                J_train.backward()
                opt.step()
                opt.zero_grad()
            model.eval()
            for batch in val_loader:
                x, y = batch
                yhat = model(x)
                J_val = criterion(yhat, y.unsqueeze(-1))
                val_acc = accuracy(yhat, y.unsqueeze(-1))
                score = F1_score(yhat, y.unsqueeze(-1))
                print(f'===============================\nEPOCH: {epoch}\nval_acc: {val_acc} \nf1_score: {score} \nval_loss: {J_val}\ntrain_loss: {J_train}\n')


for idx, model in enumerate(models):
    torch.save(model.state_dict() ,f'models/Single_LSTM_{idx+1}.pth')
torch.save(meta_classifier_1.state_dict(), 'models/LSTM_RandomNeuronalPopulation.pth')
torch.save(meta_classifier_2.state_dict(), 'models/LSTM_MetaClassifierNN.pth')

model_scripted = torch.jit.script(meta_classifier_1) # Export to TorchScript
model_scripted.save('models/LSTM_RandomNeuronalPopulation.pt') # Save


model_scripted = torch.jit.script(meta_classifier_2) # Export to TorchScript
model_scripted.save('models/LSTM_MetaClassifierNN.pt') # Save


for idx, model in enumerate(models):
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(f'models/Single_LSTM_{idx+1}.pt') # Save
