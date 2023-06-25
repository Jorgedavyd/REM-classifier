import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import binary_cross_entropy 

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


#classification module
class RemClassificationBase(nn.Module):
    def training_step(self, batch):
        input, target = batch
        out = self(input)                  # Generar predicciones
        loss = binary_cross_entropy(out, target) # Calcular el costo
        return loss

    def validation_step(self, batch):
        input, target = batch
        out = self(input)                    # Generar predicciones
        loss = binary_cross_entropy(out, target)   # Calcular el costo
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Sacar el valor expectado de todo el conjunto de costos
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result): # Seguimiento del entrenamiento
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss']))

## Model module

# Here we can change the model architecture.
def  SingularLayer(input, output):
    out = nn.Sequential(
        nn.Linear(input, output),
        nn.ReLU(True)
    )
    return out

class DeepNeuralNetwork(RemClassificationBase):
    def __init__(self, input = 3, *args):
        super(DeepNeuralNetwork, self).__init__()
        
        self.overall_structure = nn.Sequential()
        #Model input and hidden layer
        for num, output in enumerate(args):
            self.overall_structure.add_module(name = f'layer_{num+1}', module = SingularLayer(input, output))
            input = output

        #Model output layer
        self.output_layer = nn.Sequential(
                nn.Linear(input, 1),
                nn.Sigmoid()
            )
    def forward(self, xb):
        out = self.overall_structure(xb)
        out = self.output_layer(xb)
        return out

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

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