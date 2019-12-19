import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

num_epoch = 10
BATCH_SIZE = 50

transform = transforms.Compose(
    [transforms.ToTensor()
     ]
)

# CIFAR10数据集加载
train_dataset = datasets.CIFAR10(
    root='/Users/limingxia/desktop/study/AI/cifar10-CNN/cifar10_data',
    train=True,
    transform=transform,
    download=True
)
train_loader = data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_dataset = datasets.CIFAR10(
    root='/Users/limingxia/desktop/study/AI/cifar10-CNN/cifar10_data',
    train=False,
    transform=transform,
    download=False
)
test_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 搭建网络


class Net_CIFAR10(torch.nn.Module):
    def __init__(self):
        super(Net_CIFAR10, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.predict = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        #x = self.hidden(x)
        x = self.predict(x)
        return x


hidden_layer = 200

# 定义网络
net_cifar10 = Net_CIFAR10()
print(net_cifar10)

# 进行优化
optimizer = torch.optim.RMSprop(net_cifar10.parameters(), lr=0.005, alpha=0.9)
#optimizer = torch.optim.Adam(net_mnist.parameters(), lr = 0.005, betas= (0.9, 0.99))
loss_function = nn.CrossEntropyLoss()

# 开始训练
for epoch in range(num_epoch):
    print('epoch = %d' % epoch)
    for i, (batch_x, batch_y) in enumerate(train_loader):

        x = net_cifar10(batch_x)
        loss = loss_function(x, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('loss = %.5f' % loss)

# 对测试集的评估
total = 0
correct = 0
net_cifar10.eval()
for batch_x, batch_y in test_loader:
    x = net_cifar10(batch_x)
    _, prediction = torch.max(x, 1)
    total += batch_y.size(0)
    correct += (prediction == batch_y).sum()
print('There are ' + str(correct.item()) + ' correct numbers.')
print('Accuracy=%.2f' % (100.00 * correct.item() / total))
