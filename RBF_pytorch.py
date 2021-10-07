import torch
import torch.nn.functional as f
import scipy.io as scio
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from torchinfo import summary


SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def prepare_data():
    data_train_path = './data_train.mat'
    label_train_path = './label_train.mat'
    data_test_path = './data_test.mat'

    data_train = scio.loadmat(data_train_path)['data_train']
    label_train = scio.loadmat(label_train_path)['label_train'].squeeze()
    data_test = scio.loadmat(data_test_path)['data_test']
    return data_train, label_train, data_test


def getCentre(hidden_size, mode='random'):
    if mode == 'random':
        centre = np.array([data_train[i] for i in random.sample(range(data_train.shape[0]), hidden_size)])
        return centre


class RBF_NN(torch.nn.Module):

    def __init__(self, centre, hidden_size):
        """

        :param centre:
        :param data:
        """
        super().__init__()
        # centre:[hidden_size, data_dim]
        # centre = torch.tensor(centre, dtype=torch.float32).clone().detach()
        self.centre = (torch.nn.Parameter(data=torch.tensor(centre.data), requires_grad=True)).to(device)
        self.linear = torch.nn.Linear(hidden_size, 2)
        self.dropout = torch.nn.Dropout(0.5)

    def gaussianKernel(self, data):
        """

        :param data: [1, data_dim]
        :return:[1, data_dim]
        """
        # batch_size = data.size[0]
        a = self.centre
        b = data.unsqueeze(0).repeat(self.centre.data.size(0), 1)
        c = torch.exp(-(a - b).pow(2).sum(1, keepdims=False))
        return c

    def forward(self, data):
        """

        :param data: [batch_size, 1, data_dim]
        :return:
        """
        output = self.gaussianKernel(data)
        output = self.dropout(output)
        output = self.linear(output)
        # return torch.sigmoid(output)
        return torch.tanh(output)
        # return output


def train(epoch):
    # parameter setting
    s = data_train.shape[0]
    rbf.train()
    total_loss = 0
    loss_list = []
    # bar = tqdm(enumerate(zip(data_train, label_train)), total=len(label_train), ascii=True, desc='train')
    # for index, (data, label) in bar:
    for index, (data, label) in enumerate(zip(data_train, label_train)):
        # label = [1, 0] if label == 1 else [0, 1]
        label = [1, 0] if label == 1 else [0, 1]
        label = torch.tensor(label, dtype=torch.float32, requires_grad=True).to(device)
        data = torch.tensor(data, dtype=torch.float32, requires_grad=True).to(device)
        if rbf.centre.grad is not None:
            rbf.centre.grad.data.zero_()
        optimizer.zero_grad()
        output = rbf(data)

        ###################one-hot###########################
        # log_prob = torch.nn.functional.log_softmax(output, dim=0)
        # loss = -torch.sum(log_prob * label) / s
        ###################one-hot###########################

        loss = f.mse_loss(output, label)
        total_loss += loss
        loss_list.append(loss.item())
        rbf.centre.retain_grad()
        loss.backward()
        rbf.centre.data = rbf.centre.data - lr * rbf.centre.grad
        optimizer.step()
        # bar.set_description("epoch:{}\tindex:{}\t loss:{:.2f}".format(epoch, index, loss.item()))

    return total_loss


def eval():
    correct = 0
    with torch.no_grad():
        for index, (data, label) in enumerate(zip(data_train, label_train)):
            data = torch.tensor(data, dtype=torch.float32, requires_grad=True).to(device)
            predict = rbf(data)

            # val, indices = torch.max(predict, dim=-1)
            # predict = 1 if indices == 0 else -1

            predict = 1 if predict >= 0 else -1
            if predict == label:
                correct += 1
    train_acc = correct / len(label_train)
    print("acc of train=", train_acc)

    correct = 0
    with torch.no_grad():
        for index, (data, label) in enumerate(zip(data_val, label_val)):
            data = torch.tensor(data, dtype=torch.float32, requires_grad=True).to(device)
            predict = rbf(data)

            # val, indices = torch.max(predict, dim=-1)
            # predict = 1 if indices == 0 else -1

            predict = 1 if predict >= 0 else -1
            if predict == label:
                correct += 1
    valid_acc = correct / len(label_val)
    print("acc of valid=", valid_acc)
    return train_acc, valid_acc

def predict_test():
    prediction = []
    for data in data_test:
        prediction.append(rbf(data))
    ans = []
    for p in prediction:
        val, indices = torch.max(p, dim=-1)
        p = 1 if indices == 0 else -1
        if p >= 0:
            ans.append(1)
        else:
            ans.append(-1)
    return ans

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_train, label_train, data_test = prepare_data()
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, stratify=label_train,
                                                                    random_state=SEED)
    model_path = './model' + '.one-hot'
    optim_path = './optim' + '.one-hot'

    hidden_size = 81
    lr = 0.001
    centre = getCentre(hidden_size)
    centre = torch.tensor(centre, dtype=torch.float32).to(device)

    rbf = RBF_NN(centre, hidden_size).to(device)
    optimizer = torch.optim.Adam(params=rbf.parameters(), lr=lr)
    summary(rbf, input_size=(33,))
    if os.path.exists(model_path) and os.path.exists(optim_path):
        rbf.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optim_path))

    # best_valid_acc = 0
    # for epochs in range(50):
    #     print("training epochs = ", epochs)
    #
    #     # c1 = rbf.centre
    #     # print(c1)
    #     total_loss = 0
    #     bar = tqdm(range(50), desc='total_loss', ascii=True)
    #     for i in bar:
    #         total_loss = train(i)
    #         bar.set_description("epoch:{}\t total_loss:{:.2f}".format(i, total_loss))
    #
    #     # c2 = rbf.centre
    #     # print(c2)
    #
    #     train_acc, valid_acc = eval()
    #
    #     if valid_acc >= best_valid_acc:
    #         best_valid_acc = valid_acc
    #         torch.save(rbf.state_dict(), model_path)
    #         torch.save(optimizer.state_dict(), optim_path)

    data_test = torch.tensor(data_test, dtype=torch.float32, requires_grad=True).to(device)
    ans = predict_test()
    print(ans)
