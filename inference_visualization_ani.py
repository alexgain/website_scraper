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
import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int,default=-1)
parser.add_argument('--interv', type=int,default=1)
parser.add_argument('--thresh', type=float,default=0.50)
parser.add_argument('--smooth', type=int,default=1)
opt = parser.parse_args()
# path = '/anaconda3/pkgs/tensorflow-base-1.9.0-mkl_py36h70e0e9a_0/lib/python3.6/site-packages/tensorflow/contrib/ffmpeg'
# path = './ffmpeg'
# plt.rcParams['animation.ffmpeg_path'] = path

t1 = time()

load_path = './saved_models/test6.state'
save_path = '/Users/AlexGain/Google Drive (Amplio)/Website/'

## hyperparams 1:
id_ = opt.id
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
yhat = np.round(smooth(yhat,opt.smooth))

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
# line1 = ax.scatter(np.arange(A[:,0].shape[0]),A[:,0],c=yhat, cmap=matplotlib.colors.ListedColormap(colors),s=0.3)
# line2 = ax.scatter(np.arange(A[:,1].shape[0]),A[:,1],c=yhat, cmap=matplotlib.colors.ListedColormap(colors),s=0.3)
# line3 = ax.scatter(np.arange(A[:,2].shape[0]),A[:,2],c=yhat, cmap=matplotlib.colors.ListedColormap(colors),s=0.3)
plt.plot([], [], 'o', label='Jumping Intervals',color=colors[1])
# plt.plot([], [], 'o', label='Running',color=colors[0])
plt.title('Accelerometer Data')
plt.xlabel('Seconds')
plt.ylabel('milli-g')
plt.legend()
plt.savefig('./last_input.png',dpi=300,bbox_inches = "tight")
# plt.show()

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        self.ax.axis([-10, 10, -10, 10])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        xy = (np.random.random((self.numpoints, 2))-0.5)*10
        s, c = np.random.random((self.numpoints, 2)).T
        while True:
            xy += 0.03 * (np.random.random((self.numpoints, 2)) - 0.5)
            s += 0.05 * (np.random.random(self.numpoints) - 0.5)
            c += 0.02 * (np.random.random(self.numpoints) - 0.5)
            yield np.c_[xy[:,0], xy[:,1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


# fig, ax = plt.subplots()
# line1, = ax.plot(A[:,0],label='X acce.')
# line2, = ax.plot(A[:,1],label='Y acce.')
# line3, = ax.plot(A[:,2],label='Z acce.')
# plt.title('Accelerometer Data')
# plt.xlabel('Timestep')
# plt.ylabel('milli-g')
# plt.legend()
# plt.savefig('accelerometer.png',dpi=1000)

def update(num, line1,line2,lin3):
    line1.set_offsets(np.array([np.arange(0,num*opt.interv)*0.002403846,A[:,0][:num*opt.interv]]).T)
    # line1.set_color(c=yhat[:num*opt.interv:opt.interv])
    line2.set_offsets(np.array([np.arange(0,num*opt.interv)*0.002403846,A[:,1][:num*opt.interv]]).T)
    # line2.set_color(c=yhat[:num*opt.interv:opt.interv])
    line3.set_offsets(np.array([np.arange(0,num*opt.interv)*0.002403846,A[:,2][:num*opt.interv]]).T)
    # line3.set_color(c=yhat[:num*opt.interv:opt.interv])
    return line1,line2,line3

#A.shape[0]//opt.interv
ani = animation.FuncAnimation(fig, update, A.shape[0]//opt.interv, fargs=[line1,line2,line3],
                              interval=1, blit=True)#, save_count=50)

# Writer = animation.writers['ffmpeg_file']
# writer = Writer(fps=60, bitrate=100)

# writer = animation.FFMpegWriter(fps=600, metadata=dict(artist='Me'), bitrate=100)
writer = animation.ImageMagickWriter(fps=1000)

# writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=100)

ani.save('./accelerometer.gif', writer=writer)

t2 = time()
print('Time Elapsed:',(t2 - t1)/60,'minutes')


