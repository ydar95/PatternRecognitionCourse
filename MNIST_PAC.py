import torch
import torch.nn as nn
import numpy as np
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
image_src_size = 28*28

epoch_num  = 33
batch_size = 512

def get_mnist_pac(dateset , pac_size):
    pac_dataloader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=len(mnist_train), shuffle=True)
    input_data = []
    for i, (img, label) in enumerate(pac_dataloader): #只运行一次
        input_data = img.view(-1, image_src_size).cpu().numpy()
    #print(input_data.shape)
    mnist_pac = PCA(n_components=pac_size)
    mnist_pac.fit(input_data)
    return mnist_pac


datasets_img_transform = transforms.Compose([transforms.ToTensor()]) # 归一化
mnist_train = datasets.MNIST(root='./data/',train=True,download=True ,transform=datasets_img_transform) #mnist 数据集
mnist_test = datasets.MNIST(root='./data/',train=False,download=True ,transform=datasets_img_transform) #mnist 数据集

class BPNet(nn.Module):
    def __init__(self,input_dim):
        super(BPNet, self).__init__()
        self.classify= nn.Sequential(
            nn.Linear(input_dim , 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 10),
        )

    def forward(self,x):
        x=self.classify(x)
        return F.log_softmax(x ,dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size)


def train(model, device, train_loader, optimizer, epoch ,batch_size,mnist_pac=None):
    model.train()

    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if mnist_pac!=None:
            data = mnist_pac.transform(data.view(data.shape[0],-1).cpu().numpy())
            data =torch.Tensor(data)

        data, target = data.view(data.shape[0],-1).to(device), target.to(device)


        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        if(batch_idx+1)%30 == 0 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        if batch_idx == (len(train_loader) - 1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx) * batch_size +len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),loss.item()))

    #_train_loss.append(total_loss/len(train_loader.dataset))
    return total_loss/len(train_loader.dataset)


def test(model, device, test_loader,mnist_pac=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if mnist_pac != None:
                data = mnist_pac.transform(data.view(data.shape[0], -1).cpu().numpy())
                data = torch.Tensor(data)
            data, target = data.view(data.shape[0],-1).to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]                           # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'
           .format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    #_test_loss.append(test_loss)
    #_test_acc.append(correct / len(test_loader.dataset))
    return correct / len(test_loader.dataset), test_loss

train_min_loss  =[]
test_max_acc    =[]
for size in range(1,28):

    image_pac_size = size*size
    mnist_pac =get_mnist_pac(mnist_train,image_pac_size)

    model = BPNet(image_pac_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    min_loss=999.99
    max_acc=0
    for epoch in range(epoch_num):
        loss = train(model, device, train_loader, optimizer, epoch, batch_size, mnist_pac=mnist_pac)
        min_loss = min(min_loss,loss)

        acc ,_ = test(model, device, test_loader, mnist_pac=mnist_pac)
        max_acc = max(acc,max_acc)

    train_min_loss.append(min_loss)
    test_max_acc.append(max_acc)

plt.plot(range(len(train_min_loss)),train_min_loss)
plt.show()

plt.plot(range(len(test_max_acc)),test_max_acc)
plt.show()

print('train_min_loss',train_min_loss)
print('train_min_loss',test_max_acc)

# plt.plot(range(len(_train_loss)),_train_loss)
# plt.show()
#
# plt.plot(range(len(_test_loss)),_test_loss)
# plt.show()
#
# plt.plot(range(len(_test_acc)),_test_acc)
# plt.show()
#
# print(_train_loss)
# print(_test_loss)
# print(_test_acc)
