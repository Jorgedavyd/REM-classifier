import pandas as pd

def get_individual():
    id_ = [  46343,  759667,  781756,  844359, 1066528, 1360686, 1449548,
       1455390, 1818471, 2598705, 2638030, 3509524, 3997827, 4018081,
       4314139, 4426783, 5132496, 5383425, 5498603, 5797046, 6220552,
       7749105, 8000685, 8173033, 8258170, 8530312, 8686948, 8692923,
       9106476, 9618981, 9961348]
    data_list = []
    
    df = pd.read_csv('data/data.csv')
    
    for i in id_:
        data = df.loc[df['Unnamed: 0'] == i, :].drop(['Unnamed: 0', 'Unnamed: 1', 'cosine'], axis =1)
        data_list.append(data)
    
    return data_list