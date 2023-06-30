import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import binary_cross_entropy 
import numpy as np 
from sklearn.preprocessing import StandardScaler

##data preprocessing
def dataframe_to_torch(dataframe, input_cols, output_cols):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    #Normalizing the dataset
    inputs_norm = StandardScaler().fit_transform(inputs_array)
    #Creating torch tensors
    inputs = torch.from_numpy(inputs_norm.astype(np.float32()))
    targets = torch.from_numpy(targets_array.astype(np.float32()))
    return inputs, targets

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

#Accuracy metric
def accuracy(outputs, targets):
    rounded_predictions = torch.round(outputs)
    accuracy = torch.Tensor((rounded_predictions == targets).sum().item() / len(targets))
    return accuracy

#classification module
class RemClassificationBase(nn.Module):
    def training_step(self, batch):
        inputs, targets = batch
        # Reshape target tensor to match the input size
        target_tensor = targets.unsqueeze(1)  # Add a new dimension
        target_tensor = target_tensor.expand(-1, 1)  # Duplicate values across second dimension
        out = self(inputs)                  # Generar predicciones
        loss = binary_cross_entropy(out, target_tensor) # Calcular el costo
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        target_tensor = targets.unsqueeze(1)  # Add a new dimension
        target_tensor = target_tensor.expand(-1, 1)  # Duplicate values across second dimension
        out = self(inputs)                    # Generar predicciones
        loss = binary_cross_entropy(out, target_tensor)   # Calcular el costo
        acc = accuracy(out, targets) #Calcular la precisión
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()   # Sacar el valor expectado de todo el conjunto de precisión
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

## Model module

# Here we can change the model architecture.
def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    )
    return out

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

def fit(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=False, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = [] # Seguimiento de entrenamiento

    # Poner el método de minimización personalizado
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Learning rate scheduler, le da momento inicial al entrenamiento para converger con valores menores al final
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()  #Activa calcular los vectores gradiente
        train_losses = []
        lrs = [] # Seguimiento
        for batch in train_loader:
            # Calcular el costo
            loss = model.training_step(batch)
            #Seguimiento
            train_losses.append(loss)
            #Calcular las derivadas parciales
            loss.backward()

            # Gradient clipping, para que no ocurra el exploding gradient
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            #Efectuar el descensod e gradiente y borrar el historial
            optimizer.step()
            optimizer.zero_grad()

            # Guardar el learning rate utilizado en el cycle.
            lrs.append(get_lr(optimizer))
            #Utilizar el siguiente valor de learning rate dado OneCycle scheduler
            sched.step()

        # Fase de validación
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item() #Stackea todos los costos de las iteraciones sobre los batches y los guarda como la pérdida general de la época
        result['lrs'] = lrs #Guarda la lista de learning rates de cada batch
        model.epoch_end(epoch, result) #imprimir en pantalla el seguimiento
        history.append(result) # añadir a la lista el diccionario de resultados
    return history
