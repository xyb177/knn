import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

np.set_printoptions(suppress = True)
def standardization (X):
    maxn = X.max(axis=0)
    minn = X.min(axis=0)
    return (X -minn) /(maxn-minn)
def load_data():
    data = pd.read_csv("housing.csv",header = None)
    row = data.shape[0]
    col = len(data.iloc[0,0].split())
    dataset = np.empty([row,col])

    for i in range(row):
        temp = data.iloc[i,0].split()
        dataset[i] = np.array(temp[:])
    
    train_set_size = int(row*0.8)
    test_set_size = row-train_set_size
    train_set = dataset[:train_set_size]
    test_set = dataset[train_set_size:]
   
   
    X_train = train_set[:,:-1]
    Y_train = train_set[:,-1]
    Y_train.reshape(-1,1)
    print(Y_train.shape)
    X_test = test_set[:,:-1]
    Y_test = test_set[:,-1]
    return X_train,Y_train,X_test,Y_test


class linar_regression():
    def fitness(self,X_train,Y_train, learning_rate=0.5, lamda = 0.03):
        m,n = X_train.shape
        X = np.c_[X_train, np.ones(m)]
        self.w = np.zeros([n+1,1])
        print(X.shape , self.w.shape)
        max_cnt = int(1e8)
        last_better = 0  # 上一次得到较好学习误差的迭代学习次数
        last_Jerr = int(1e8)  # 上一次得到较好学习误差的误差函数值
        threshold_value = 1e-8  # 定义在得到较好学习误差之后截止学习的阈值
        threshold_count = 10  #
        for i in range(max_cnt):
            loss = np.sum((X.dot(self.w)-Y_train) **2) /(2*m)
            predict = X.dot(self.w)
          
            print(predict.shape)
            self.w  = self.w-learning_rate * ((X.T @ (X.dot(self.w)-Y_train)) / m + lamda * self.w)
            print(self.w.shape)
            if loss < last_Jerr - threshold_value:          # 检测损失函数的变化值，提前结束迭代
                last_Jerr = loss
                last_better = i
            elif i - last_better > threshold_count:
                break    

    def predict(self, X_test):
        num  = X_test.shape[0]
        X = np.c_[X_test, np.ones(num,)]
        print(X.shape)
        print(self.w.shape)
        print(X)
        print(self.w)
        Y = X.dot(self.w)
        return  Y       


if __name__ == "__main__":
    X_train,Y_train,X_test,Y_test = load_data()
    
    model = linar_regression()
    model.fitness(X_train,Y_train)
    Y = model.predict(X_test)
    t = np.arange(len(Y))
    print(Y,Y_test)
    plt.figure(facecolor='w')
    plt.plot(t, Y_test, 'c-', lw=1.6, label=u'actual value')
    plt.plot(t, Y, 'm-', lw=1.6, label=u'estimated value')
    plt.legend(loc='best')
    plt.title(u'Boston house price', fontsize=18)
    plt.xlabel(u'case id', fontsize=15)
    plt.ylabel(u'house price', fontsize=15)
    plt.grid()
    plt.show()


