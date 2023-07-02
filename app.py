import torch
from model.utils import scaler, transform
import pandas as pd

model = torch.jit.load('model\model_hr_motion.pt')
model.eval()


# Data importation

dataframe = pd.read_csv('model/data/data.csv')

#Getting the columns
columns = dataframe.columns.values

#preprocessing
#scaler
input_cols = columns[[2,3,5]]

scaler = scaler(dataframe, input_cols)

#Preprocessing
df = dataframe.copy(deep=True)

# Extract input & outupts as numpy arrays
data = df[input_cols].to_numpy()


transform = transform(dataframe, scaler)

##Program structure

while True:
    ##Import real-time state
    state = transform(torch.tensor())
    ##Evaluate state
    if model(state): #One hot encoded answer.
        print('REM')
        continue
    else:
        pass