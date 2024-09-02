import torch 
import numpy as np
import os 
import gzip 

def load_data(file_path):
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    
    paths = []
    for file in files:
        paths.append(os.path.join(file_path,file))
    
    with gzip.open(paths[0],'rb') as train_label_path:
        y_train = np.frombuffer(train_label_path.read(),np.uint8,offset=8) 
    with gzip.open(paths[1],'rb') as train_image_path:
        x_train = np.frombuffer(train_image_path.read(),np.uint8,offset=16)  
    with gzip.open(paths[2],'rb') as test_label_path:
        y_test = np.frombuffer(test_label_path.read(),np.uint8,offset=8)
    with gzip.open(paths[3],'rb') as test_image_path:
        x_test = np.frombuffer(test_image_path.read(),np.uint8,offset=16)  
    return (x_train,y_train),(x_test,y_test)  

if __name__ == "__main__":
    torch.manual_seed(0)