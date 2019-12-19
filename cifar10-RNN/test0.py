import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

batch_size = 384
train_dataset = torchvision.datasets.CIFAR10(root='/Users/limingxia/desktop/study/AI/cifar10-CNN/cifar10_data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root='/Users/limingxia/desktop/study/AI/cifar10-CNN/cifar10_data',
                                            train=False,
                                            transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# ---------------------
# Exploring the dataset
# ---------------------
# function to sbow an image


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
# show image
imshow(torchvision.utils.make_grid(images, nrow=15))

# ----------
# parameters
# ----------
N_STEPS = 32
N_INPUTS = 32  # 输入数据的维度
N_NEURONS = 100  # RNN中间的特征的大小
N_OUTPUT = 10  # 输出数据的维度(分类的个数)
N_EPHOCS = 10  # epoch的大小
N_LAYERS = 3

# ------
# models
# ------


class ImageRNN(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons, n_outputs, n_layers):
        super(ImageRNN, self).__init__()
        self.batch_size = batch_size  # 输入的时候batch_size
        self.n_inputs = n_inputs  # 输入的维度
        print(self.n_inputs)
        self.n_outputs = n_outputs  # 分类的大小
        self.n_neurons = n_neurons  # RNN中输出的维度
        print(self.n_neurons)
        self.n_layers = n_layers  # RNN中的层数

        self.basic_rnn = nn.RNN(
            self.n_inputs, self.n_neurons, num_layers=self.n_layers)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self):
        # (num_layers, batch_size, n_neurons)
        # initialize hidden weights with zero values
        # 这个是net的memory, 初始化memory为0
        return (torch.zeros(self.n_layers, self.batch_size, self.n_neurons).to(device))

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = x.view(x.size(0), -1)
        #x = self.hidden(x)
        #x = self.predict(x)
        # transforms x to dimensions : n_step × batch_size × n_inputs
        x = x.permute(1, 0, 2)  # 需要把n_step放在第一个
        # 每次需要重新计算batch_size, 因为可能会出现不能完整方下一个batch的情况
        self.batch_size = x.size(1)
        self.hidden = self.init_hidden()  # 初始化hidden state
        rnn_out, self.hidden = self.basic_rnn(x, self.hidden)  # 前向传播
        out = self.FC(rnn_out[-1])  # 求出每一类的概率
        return out.view(-1, self.n_outputs)  # 最终输出大小 : batch_size X n_output


# --------------------
# Device configuration
# --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------
# Test the model(输入一张图片查看输出)
# ------------------------------------
# 定义模型
model = ImageRNN(batch_size, N_INPUTS, N_NEURONS,
                 N_OUTPUT, N_LAYERS).to(device)
# 初始化模型的weight
model.basic_rnn.weight_hh_l0.data = torch.eye(
    n=N_NEURONS, m=N_NEURONS, out=None).to(device)
model.basic_rnn.weight_hh_l1.data = torch.eye(
    n=N_NEURONS, m=N_NEURONS, out=None).to(device)
model.basic_rnn.weight_hh_l2.data = torch.eye(
    n=N_NEURONS, m=N_NEURONS, out=None).to(device)
# 定义数据
dataiter = iter(train_loader)
images, labels = dataiter.next()
model.hidden = model.init_hidden()
#logits = model(images.view(-1, 32, 32).to(device))
# print(logits[0:2])
"""
tensor([[-0.2846, -0.1503, -0.1593,  0.5478,  0.6827,  0.3489, -0.2989,  0.4575,
         -0.2426, -0.0464],
        [-0.6708, -0.3025, -0.0205,  0.2242,  0.8470,  0.2654, -0.0381,  0.6646,
         -0.4479,  0.2523]], device='cuda:0', grad_fn=<SliceBackward>)
"""

# 产生对角线是1的矩阵
torch.eye(n=5, m=5, out=None)
"""
tensor([[1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1.]])
"""

# --------
# Training
# --------
model = ImageRNN(batch_size, N_INPUTS, N_NEURONS,
                 N_OUTPUT, N_LAYERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 初始化模型的weight
model.basic_rnn.weight_hh_l0.data = torch.eye(
    n=N_NEURONS, m=N_NEURONS, out=None).to(device)
model.basic_rnn.weight_hh_l1.data = torch.eye(
    n=N_NEURONS, m=N_NEURONS, out=None).to(device)
model.basic_rnn.weight_hh_l2.data = torch.eye(
    n=N_NEURONS, m=N_NEURONS, out=None).to(device)


def get_accuracy(logit, target, batch_size):
    """最后用来计算模型的准确率
    """
    corrects = (torch.max(logit, 1)[1].view(
        target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


# ---------
# 开始训练
# ---------
for epoch in range(N_EPHOCS):
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()
    loss = 0.0
    # trainging round
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        # reset hidden states
        model.hidden = model.init_hidden()
        # get inputs
        inputs, labels = data
        print(batch_size)
        inputs = inputs.view(-1, 32, 32).to(device)
        labels = labels.to(device)
        # forward+backward+optimize
        outputs = model(inputs)
        loss += criterion(outputs, labels).item()
        loss.backward()
        optimizer.step()
        train_running_loss = train_running_loss + loss.detach().item()
        train_acc = train_acc + get_accuracy(outputs, labels, batch_size)
    model.eval()
    print('Epoch : {:0>2d} | Loss : {:<6.4f} | Train Accuracy : {:<6.2f}%'.format(
        epoch, train_running_loss / i, train_acc / i))

# ----------------------------------------
# Computer accuracy on the testing dataset
# ----------------------------------------
test_acc = 0.0
for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    labels = labels.to(device)
    inputs = inputs.view(384, 32, 32).to(device)
    outputs = model(inputs)
    thisBatchAcc = get_accuracy(outputs, labels, batch_size)
    print("Batch:{:0>2d}, Accuracy : {:<6.4f}%".format(i, thisBatchAcc))
    test_acc = test_acc + thisBatchAcc
print('============平均准确率===========')
print('Test Accuracy : {:<6.4f}%'.format(test_acc/i))
