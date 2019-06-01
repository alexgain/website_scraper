import numpy as np
import matplotlib.pyplot as plt
import copy
from time import time
from models import *
from utils.sampler import ImbalancedDatasetSampler, smooth
from get_email_csvs import return_last_csv
import matplotlib.animation as animation

# loading
load_path = './saved_models/test2.state'
save_path = './saved_models/test6.state'

## hyperparams:
smoothing = 1
test_perc = 0.02
width = 500

epochs = 100
BS = 128
LR = 0.005        
momentum = 0.9
# id_ = -1


## loading dataset:
# data = np.concatenate([return_last_csv(id_ = -4),return_last_csv(id_ = -3),return_last_csv(id_ = -2)[:10000]],axis=0)
# data = data[:,:6]
data = return_last_csv(id_ = -4)
data = data[:23650,:6]
# for j in range(data.shape[1]):
#     data[:,j] -= data[:,j].min()
#     data[:,j] /= data[:,j].max()
for j in range(data.shape[1]):
    data[:,j] /= 1e4
    data[:,j] += 3.4
    data[:,j] = smooth(data[:,j],smoothing)
    

# data = np.concatenate((data[:3000],data[4000:],data[4000:],data[4000:5500]),axis=0)
# data = np.concatenate((data[:3000],data[4000:]),axis=0)
y = np.zeros(data.shape[0])

## manually assigning labels:

# y[12100:14700] += 1
# y[23270:26150] += 1

y[12200:12400] += 1
y[12735:12970] += 1
y[13635:13840] += 1
y[14230:14630] += 1
y[23380:23600] += 1

y = y.reshape(-1,1)
data = np.concatenate((data,y),axis=1)

# plt.hist(y)
# stop

## creating test set:
np.random.shuffle(data)
N = int(round((1-test_perc)*data.shape[0]))
xtrain = torch.Tensor(data[:N,:data.shape[1]-1])
ytrain = torch.Tensor(data[:N,-1]).long()
xtest = torch.Tensor(data[N:,:data.shape[1]-1])
ytest = torch.Tensor(data[N:,-1]).long()

## data_loaders:
train = torch.utils.data.TensorDataset(xtrain, ytrain)
test = torch.utils.data.TensorDataset(xtest, ytest)

# train_loader = torch.utils.data.DataLoader(train, batch_size=BS, sampler = ImbalancedDatasetSampler(train,hand_labels=ytrain.cpu().data.numpy()))
train_loader = torch.utils.data.DataLoader(train, batch_size=BS)
test_loader = torch.utils.data.DataLoader(test, batch_size=BS, shuffle=False)

## defining model and optimizer:
my_net = MLP(xtrain.shape[1],width=width,num_classes=np.unique(y).shape[0])
loss_metric = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(my_net.parameters(), lr = LR, momentum = momentum)
optimizer = torch.optim.Adam(my_net.parameters(), lr = LR)

## Train/test bolier plate code:
def train_acc(verbose = 1):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in train_loader:            
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        loss_sum += loss_metric(outputs,Variable(labels)).cpu().data.numpy().item()
        
    if verbose:
        print('Accuracy of the network on the train data: %f %%' % (100.0 * np.float(correct) / np.float(total)))

    return 100.0 * np.float(correct) / np.float(total), loss_sum
    
def test_acc(verbose = 1, flatten=False, input_shape = 28*28):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in test_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        loss_sum += loss_metric(outputs,Variable(labels)).cpu().data.numpy().item()

    if verbose:
        print('Accuracy of the network on the test data: %f %%' % (100.0 * np.float(correct) / np.float(total)))

    return 100.0 * np.float(correct) / np.float(total), loss_sum

t1 = time()
for epoch in range(epochs):

    ##time-keeping 1:
    time1 = time()

    for i, (x,y) in enumerate(train_loader):
        
        ##cuda:
        if gpu_boole:
            x = x.cuda()
            y = y.cuda()

        ##data preprocessing for optimization purposes:        
        x = Variable(x)
        y = Variable(y)

        ###regular BP gradient update:
        optimizer.zero_grad()
        outputs = my_net.forward(x)
        loss = loss_metric(outputs,y)# - 0.1*bap_test
        loss.backward()
                
        ##performing update:
        optimizer.step()

    print ('Epoch [%d/%d], Train Loss: %.4f' 
           %(epoch+1, epochs, loss.data.item()))
    
    train_perc, loss_train = train_acc()
    test_perc, loss_test = test_acc()


    time2 = time()

    print('Elapsed time for epoch:',time2 - time1,'s')
    print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
    print()


plt.plot(torch.argmax(my_net(xtest),dim=1).cpu().data.numpy(),'-o',label='prediction')
plt.plot(ytest.cpu().data.numpy(),'-o',label='ground truth')
plt.xlabel('timestep')
plt.ylabel('class')
plt.legend()
plt.show()

plt.plot(torch.argmax(my_net(xtest),dim=1).cpu().data.numpy()!=ytest.cpu().data.numpy(),'o',label='incorrect steps')
plt.xlabel('timestep')
plt.ylabel('class')
plt.legend()
plt.show()


# torch.save(my_net.state_dict(), save_path)
