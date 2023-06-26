import torch

model = torch.jit.load('model.pt')
model.eval()

##Program structure

while True:
    ##Import real-time state
    state = torch.tensor()
    ##Evaluate state
    if model(state): #One hot encoded answer.
        ##Send notification