import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
from time import time
from models import *
from utils.sampler import ImbalancedDatasetSampler, smooth
from get_email_csvs import return_last_csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

load_path = './saved_models/test6.state'
save_path = '/Users/AlexGain/Google Drive (Amplio)/Website/'

## hyperparams 1:
id_ = -1
thresh = 0.50
smoothing = 1

## hyperparams 2:
test_perc = 0.3
width = 500
num_classes = 2

## hyperparams 3:
epochs = 30
BS = 128
LR = 0.005        
momentum = 0.9


## loading dataset:
data = return_last_csv(id_ = id_)
data = data[:,:6]
for j in range(data.shape[1]):
    # data[:,j] -= data[:,j].min()
    # data[:,j] /= data[:,j].max()
    data[:,j] /= 1e4
    data[:,j] += 3.4
    # data[:,j] = smooth(data[:,j],100)

data = torch.Tensor(data)

## defining model and optimizer:
my_net = MLP(data.shape[1],width=width,num_classes=num_classes)
my_net.load_state_dict(torch.load(load_path))

## inference
yhat = nn.Softmax()(my_net(data))[:,1]#.argmax(dim=1).cpu().data.numpy()
yhat = (yhat > thresh)*1
yhat = yhat.cpu().data.numpy()
yhat = np.round(smooth(yhat,75))

# yhat = nn.Softmax()(my_net(data)).cpu().data.numpy()
# yhat = smooth(yhat[:,1],smoothing) > thresh
# yhat[0] = not yhat[0]
# yhat = yhat*1

# yhat = np.round(smooth(yhat*20,10))
colors = ['teal','orange']

## visualizing output:
data = return_last_csv(id_ = id_)
data = data[:,:6]
plt.style.use('dark_background')
A = data[:,0:3]*0.488 #Accelerometer (x,y,z), milli-g
G = data[:,3:6]*70 #Gyroscope (x,y,z), mDPS
# A = G

#2D plot accelerometer:
fig, ax = plt.subplots()
line1 = ax.scatter(np.arange(A[:,0].shape[0])*0.002403846,A[:,0],c=yhat, cmap=matplotlib.colors.ListedColormap(colors),s=0.3)
line2 = ax.scatter(np.arange(A[:,1].shape[0])*0.002403846,A[:,1],c=yhat, cmap=matplotlib.colors.ListedColormap(colors),s=0.3)
line3 = ax.scatter(np.arange(A[:,2].shape[0])*0.002403846,A[:,2],c=yhat, cmap=matplotlib.colors.ListedColormap(colors),s=0.3)
plt.plot([], [], 'o', label='Jumping Intervals',color=colors[1])
# plt.plot([], [], 'o', label='Running',color=colors[0])
plt.title('Accelerometer Data')
plt.xlabel('Seconds')
plt.ylabel('milli-g')
plt.legend()
plt.savefig(save_path+'last_input.png',dpi=100)
# plt.show()

# interv = 50
# def update(num, line1,line2,lin33):
#     line1.set_array(np.array([np.arange(0,num*interv,interv),A[:,0][:num*interv:interv]]).T)
#     line2.set_array(np.array([np.arange(0,num*interv,interv),A[:,1][:num*interv:interv]]).T)
#     line3.set_array(np.array([np.arange(0,num*interv,interv),A[:,2][:num*interv:interv]]).T)
#     return line1,line2,line3

# ani = animation.FuncAnimation(fig, update, A.shape[0]//interv, fargs=[line1,line2,line3],
#                               interval=1, blit=True, save_count=50)

# Writer = animation.writers['pillow']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=18000)

# ani.save(save_path+'last_input.gif', writer=writer)


#2D plot accelerometer:
# fig, ax = plt.subplots()
# line1 = ax.scatter(np.arange(A[:,0].shape[0]),smooth(yhat*20,3),c=yhat, cmap=matplotlib.colors.ListedColormap(colors),s=0.3)
# plt.plot([], [], 'o', label='Box Jumps',color=colors[1])
# # plt.plot([], [], 'o', label='Running',color=colors[0])
# plt.title('Box Jumps Classification')
# plt.xlabel('Timestep')
# plt.ylabel('Box Jump Likelihood')
# plt.legend()
# plt.savefig(save_path+'last_input2.png',dpi=1000)
# plt.show()


# Nstart, Nend = 12200,12400
# Nstart, Nend = 12735,12970
# Nstart, Nend = 13635,13840
# Nstart, Nend = 14230,14630
# Nstart, Nend = 23380,23600

# Nstart, Nend = 23270,26150
# Nstart, Nend = 23270,26150
# fig, ax = plt.subplots()
# line1 = ax.scatter(np.arange(A[Nstart:Nend,0].shape[0]),A[Nstart:Nend,0],s=0.3)
# line2 = ax.scatter(np.arange(A[Nstart:Nend,1].shape[0]),A[Nstart:Nend,1],s=0.3)
# line3 = ax.scatter(np.arange(A[Nstart:Nend,2].shape[0]),A[Nstart:Nend,2],s=0.3)
# plt.plot([], [], 'o', label='Box Jumps',color=colors[1])
# plt.plot([], [], 'o', label='Running',color=colors[0])
# plt.title('Accelerometer Data')
# plt.xlabel('Timestep')
# plt.ylabel('milli-g')
# plt.legend()
# plt.show()



