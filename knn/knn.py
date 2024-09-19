import torch 
import numpy as np
import os 
import gzip 

#从本地文件读取数据集
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
        x_train = np.frombuffer(train_image_path.read(),np.uint8,offset=16).reshape(len(y_train),1,-1)
    with gzip.open(paths[2],'rb') as test_label_path:
        y_test = np.frombuffer(test_label_path.read(),np.uint8,offset=8)
    with gzip.open(paths[3],'rb') as test_image_path:
        x_test = np.frombuffer(test_image_path.read(),np.uint8,offset=16).reshape(len(y_test),1,-1)
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
        temp = np.sum(dist**2,axis=1)
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
    (train_images, train_labels), (test_images, test_labels) = load_data('MNIST/')
    sum = 0
    true_num = 0
    for i in range(len(test_images)):
        sum += 1
        result = predict(train_images,train_labels,test_images[i],10)
        print(result)
        if result == test_labels[i] :
            true_num += 1
    print(true_num/sum)