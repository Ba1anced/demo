from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
import scipy.io as scio
# import pandas as pd
import matplotlib.pyplot as plt

SEED = 10


def prepare_data():
    data_train_path = './data_train.mat'
    label_train_path = './label_train.mat'
    data_test_path = './data_test.mat'

    data_train = scio.loadmat(data_train_path)['data_train']
    label_train = scio.loadmat(label_train_path)['label_train'].squeeze()
    data_test = scio.loadmat(data_test_path)['data_test']
    return data_train, label_train, data_test

def train(c, gamma):
    #
    data_train, label_train, data_test = prepare_data()
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train,
                                                                    stratify=label_train, random_state=SEED)
    classifier = SVC(C=c, gamma=gamma*0.1, kernel='rbf')
    classifier.fit(data_train, label_train)

    predict_val = classifier.predict(data_val)
    temp = (predict_val == label_val)
    acc = sum(temp)/len(temp)

    print("c={}\tgamma={}\tacc={}".format(c, gamma, acc))

    predict_test = classifier.predict(data_test)
    print(predict_test)
    return acc


if __name__ == '__main__':
    acc_list = []
    for c in range(1, 5):
        for gamma in range(1, 10):
            acc = train(c, gamma)
            acc_list.append(acc)

    print(acc_list)

    train(c=2, gamma=1)
