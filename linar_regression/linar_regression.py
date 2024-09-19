import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

def load_data():
    data = pd.read_csv("housing.csv",header = None)
    row = data.shape[0]
    col = len(data.iloc[0,0].split())
    dataset = np.empty([row,col])

    for i in range(row):
        temp = data.iloc[i,0].split()
        print(temp)
        dataset[i] = np.array(temp[:])
    training_set_size = int(row*0.8)
    test_set_size = row-training_set_size

    
    X_train = dataset[:,:-1]
    


if __name__ == "__main__":
    data = load_data()
    print(data)


