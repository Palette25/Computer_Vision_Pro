import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import numpy as np

# torch.manual_seed(1)

# Enlarge EPOCH to 20
EPOCH = 10
LR = 0.0002
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
print(train_data.train_data.shape)

train_x = torch.unsqueeze(train_data.train_data, dim=1).type(torch.FloatTensor) / 255.
train_y = train_data.train_labels
print(train_x.shape)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255.  # Tensor on GPU
test_y = test_data.test_labels


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Two Layer CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        
        # ResNet Blocks
        self.res1 = Residual(16, 16, True)
        self.res2 = Residual(32, 32, True)

        # VGG Blocks
        self.vgg1 = vgg_block(2, 16, 32)
        self.vgg2 = vgg_block(2, 32, 16)

        # Full-Connected Network
        self.fc = nn.Linear(32, 10)
       
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2,2)) # Max Pooling with pooling kernel size 2X2
        x = self.res1(x)

        x = self.vgg1(x)
        x = self.vgg2(x)
        
        x = F.relu(self.conv2(x))  
        x = F.max_pool2d(x, (2,2)) # Max Pooling with pooling kernel size 2X2

        x = self.res2(x)
        
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = F.relu(self.fc(x))
        output = F.log_softmax(x, dim=1)
        
        return output


print(torch.cuda.is_available())


# Use multimodels for ensambling
net = Net()

device = torch.device('cuda:0')
net.to(device)

print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=LR)

#loss_func = nn.MSELoss()
loss_func = nn.CrossEntropyLoss()

# Enlarge dataSize and BatchSize
data_size = 60000
batch_size = 100

max_accuracy = 0
best_params = []

for epoch in range(EPOCH):
    random_indx = np.random.permutation(data_size)
    for batch_i in range(data_size // batch_size):
        indx = random_indx[batch_i * batch_size:(batch_i + 1) * batch_size]

        b_x = train_x[indx, :].to(device)
        b_y = train_y[indx].to(device)

        output = net(b_x)
            
        loss = loss_func(output, b_y)
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        if batch_i % 100 == 0:
            test_output = net(test_x.to(device))
            
            # Use voting mechanism to deicide final predicted label
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = torch.sum(pred_y == test_y.to(device)).type(torch.FloatTensor) / test_y.size(0)
            
            print('Epoch: ', epoch, '| train loss: get_ipython().run_line_magic(".4f'", " % loss.data.cpu().numpy(), '| test accuracy: %.4f' % accuracy)")
            if(accuracy > max_accuracy): 
                max_accuracy = accuracy
                torch.save(net, 'best_net.pkl')

test_output = net(test_x[:10].to(device))
pred_y = torch.max(test_output, 1)[1].data.squeeze()  # move the computation in GPU

print('Max Accuracy: get_ipython().run_line_magic(".4f'", " % max_accuracy)")
