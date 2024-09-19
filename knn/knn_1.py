import torch 
import numpy as np
import os 
import gzip 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import threading
from tqdm import tqdm


def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image-mean) / std
#从本地文件读取数据集
def load_data():
    train_dataset = datasets.MNIST(root='/MNIST',train=True,transform=transforms.ToTensor(),download=True)
    test_dataset = datasets.MNIST(root='/MNIST',train=False,transform=transforms.ToTensor(),download=True)

    train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size = 100, shuffle = False)
    test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size = 100, shuffle = False)
    
    x_train = train_loader.dataset.train_data.numpy()
    x_train = x_train.reshape(x_train.shape[0],28*28)
    x_train = standardize_image(x_train)
    y_train = train_loader.dataset.train_labels.numpy()

    x_test = test_loader.dataset.test_data[:10000].numpy()
    x_test = x_test.reshape(x_test.shape[0],28*28)
    x_test = standardize_image(x_test)
    y_test = test_loader.dataset.test_labels[:10000].numpy()
   
    return (x_train,y_train),(x_test,y_test)  


#KNN算法实现
def predict(train_images,train_labels,test_image,k):
    #变量n用来记录训练集具体个数
    n = len(train_labels)
    #map用来记录测试集与训练集之间的距离与相对应的训练集标签，{距离:标签,...}
    map = {}
    #distance用来记录单个测试集与所有训练集之间的距离，初始距离为0
    distance = np.zeros(n)
    
    #第一步：计算测试样本与训练数据集中样本特征之间的欧式距离
    for cnt in range(n):
        dist = test_image - train_images[cnt]
        temp = np.sum(dist**2)
        distance[cnt] = temp**0.5
        map[distance[cnt]] = train_labels[cnt]

    
    #第二步：进行递增排序
    sorted_dist = np.sort(distance)
       

    label_map = {}
    #第三步：选取前k个距离，并记录对应标签的个数
    for j in range(k):
        label = map.get(sorted_dist[j],-1)
        if label != -1 :
            label_map[label] = label_map.get(label,0) + 1

    #第四步：选取出现次数最多的标签作为预测结果
    maxn = 0
    ans = -1
    for key,value in label_map.items():
        if value > maxn :
            maxn = value
            ans = key

    return ans


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_data()
    sum = 0
    true_num = 0

    
    for i in tqdm(range(len(test_images))):
        sum += 1
        result = predict(train_images,train_labels,test_images[i],5)

        if result == test_labels[i] :
            true_num += 1
    print(true_num/sum)