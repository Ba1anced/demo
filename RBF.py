from math import exp
import scipy.io as scio
import numpy as np
import random
from scipy.linalg import norm, pinv
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from minisom import MiniSom
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SEED = 10
random.seed(SEED)
np.random.seed(SEED)


def prepare_data():
    """
    prepare data

    Returns:
        [type]: [description]
    """
    data_train_path = './data_train.mat'
    label_train_path = './label_train.mat'
    data_test_path = './data_test.mat'

    data_train = scio.loadmat(data_train_path)['data_train']
    label_train = scio.loadmat(label_train_path)['label_train'].squeeze()
    data_test = scio.loadmat(data_test_path)['data_test']
    return data_train, label_train, data_test


class RBF:
    def __init__(self, data, label, hidden_size, epsilon, centre_method='random'):
        """初始化RBF参数

        Args:
            data ([type]): 输入数据
            label ([type]): 输入数据的标签
            hidden_size ([type]): [description]
            epsilon ([type]): 如果是Gaussian函数，为函数的epsilon值
            centre_method (str, optional): 获取center的方式. Defaults to 'random'.
        """
        self.data = data
        self.label = label
        self.input_dim = data.shape[1]
        self.label_dim = label.shape[0]
        self.hidden_size = hidden_size
        self.centre = self._getCentre(hidden_size, centre_method)
        self.epsilon = epsilon
        self.weight = np.random.random((hidden_size, self.label_dim))

    def _getCentre(self, hidden_size, centre_method):
        """获取center

        Args:
            hidden_size ([type]): RBF hidden_size的大小
            centre_method ([type]): center方式，分为 'random', 'k-means', 'SOM'三种方式

        Returns:
            [type]: 返回获得的center
        """
        if centre_method == 'random':
            centre = np.array([self.data[i] for i in random.sample(range(self.data.shape[0]), hidden_size)])
            return centre
        if centre_method == 'k-means':
            centre = getKmeanscentre(hidden_size, self.data)
            return centre
        if centre_method == 'SOM':
            centre = getSOMcentre(self.hidden_size, self.input_dim, self.data)
            return centre

    def _gaussianFunc(self, centre, data):
        """返回Gaussian函数计算结果

        Args:
            centre ([type]): [description]
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        return exp(-(norm(centre - data) ** 2) / (2 * self.epsilon))

    def _calcAct(self, data):
        """计算每一个data_index, 经过hidden neuron的输出
            （可以将其看为某种激活层）
        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        G = np.zeros((data.shape[0], self.hidden_size), float)
        for centre_index, centre_i in enumerate(self.centre):
            for data_index, data_i in enumerate(data):
                G[data_index, centre_index] = self._gaussianFunc(centre_i, data_i)
        return G

    def train(self, data, label):
        """通过pinv()求伪逆矩阵， 计算从hidden_neuron输出到最后输出的weight

        Args:
            data ([type]): [description]
            label ([type]): [description]
        """
        G = self._calcAct(data)
        self.weight = np.dot(pinv(G), label)

    def eval(self, data):
        """正向传递并预测

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        G = self._calcAct(data)
        predict = np.dot(G, self.weight)
        return predict


def getSOMcentre(hidden_size, input_dim, data_train):
    som = MiniSom(1, hidden_size, input_dim, sigma=0.5, learning_rate=0.5, neighborhood_function='gaussian',
                  random_seed=10)
    som.train(data_train, num_iteration=45000)
    # winner_coordinates = np.array([som.winner(x) for x in data_train]).T
    # cluster_index = np.ravel_multi_index(winner_coordinates, (1, 10))
    # print(np.unique(cluster_index))
    # print(som.get_weights().shape)
    return som.get_weights().squeeze()


def getKmeanscentre(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    kmeans.fit(data)
    # print(kmeans.cluster_centers_.shape)
    return kmeans.cluster_centers_


def cal_acc(predict, label):
    predict_list = []
    for i in predict:
        if i > 0:
            predict_list += [1]
        else:
            predict_list += [-1]

    acc1 = []
    for pre, true in zip(predict_list, label):
        if pre == true:
            acc1 += [1]
        else:
            acc1 += [0]
    acc = sum(acc1) / (len(acc1))
    print("acc = ", acc)
    return acc


def printtest(predict):
    ans = []
    for i in predict:
        if i >= 0:
            ans += [1]
        else:
            ans += [-1]
    print(ans)


def initRBF(num_n, sig):
    print("num_n={}\tsig={}".format(num_n, sig))
    rbf = RBF(data_train, label_train, num_n, sig, centre_method='SOM')
    # print(rbf.centre)
    rbf.train(data_train, label_train)
    pre_train = rbf.eval(data_train)
    acc_train = cal_acc(pre_train, label_train)
    acc_train_list.append(acc_train)
    predict = rbf.eval(data_val)
    acc_valid = cal_acc(predict, label_val)
    acc_valid_list.append(acc_valid)
    return rbf


if __name__ == '__main__':
    data_train, label_train, data_test = prepare_data()

    # 将train_dataset分为测试集和验证集
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train,
                                                                    stratify=label_train, random_state=SEED)

    acc_train = 0
    acc_valid = 0
    acc_train_list = []
    acc_valid_list = []
    for sig in range(1, 7):
        initRBF(num_n=80, sig=sig)
    fig = plt.figure(num=1, figsize=(4, 4))
    ax = plt.subplot(111)
    ax.plot(np.arange(1, 7), acc_valid_list)
    ax.set_xticks(np.linspace(1, 7, dtype=int))
    ax.set_xlabel('sigma')
    ax.set_ylabel('accuracy')
    plt.show()
    # predict = rbf.eval(data_test)
    # printtest(predict)
