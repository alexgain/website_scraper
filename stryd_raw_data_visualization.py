import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

plt.style.use('dark_background')
# plt.style.use('fivethirtyeight')

csvpath = '../data/data_test1.csv'
data = np.loadtxt(csvpath,delimiter=',')

# N=5000
A = data[4000:,0:3]*0.488 #Accelerometer (x,y,z), milli-g
G = data[4000:,3:6]*70 #Gyroscope (x,y,z), mDPS
# A = G

#2D plot accelerometer:
fig, ax = plt.subplots()
line1, = ax.plot(A[:,0],label='X acce.')
line2, = ax.plot(A[:,1],label='Y acce.')
line3, = ax.plot(A[:,2],label='Z acce.')
plt.title('Accelerometer Data')
plt.xlabel('Timestep')
plt.ylabel('milli-g')
plt.legend()
plt.savefig('accelerometer.png',dpi=1000)

interv = 50
def update(num, line1,line2,lin33):
    line1.set_data(np.arange(0,num*interv,interv),A[:,0][:num*interv:interv])
    line2.set_data(np.arange(0,num*interv,interv),A[:,1][:num*interv:interv])
    line3.set_data(np.arange(0,num*interv,interv),A[:,2][:num*interv:interv])
    return line1,line2,line3

ani = animation.FuncAnimation(fig, update, A.shape[0]//interv, fargs=[line1,line2,line3],
                              interval=1, blit=True, save_count=50)

Writer = animation.writers['pillow']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=18000)

ani.save('./accelerometer.gif', writer=writer)

#2D plot gyroscope:
fig, ax = plt.subplots()
line1, = plt.plot(G[:,0],label='X gyro.')
line2, = plt.plot(G[:,1],label='Y gyro.')
line3, = plt.plot(G[:,2],label='Z gyro.')
plt.title('Gyroscope Data')
plt.xlabel('Timestep')
plt.ylabel('mDPS')
plt.legend()
plt.savefig('gyroscope.png',dpi=1000)

interv = 50
def update(num, line1,line2,lin33):
    line1.set_data(np.arange(0,num*interv,interv),G[:,0][:num*interv:interv])
    line2.set_data(np.arange(0,num*interv,interv),G[:,1][:num*interv:interv])
    line3.set_data(np.arange(0,num*interv,interv),G[:,2][:num*interv:interv])
    return line1,line2,line3

ani = animation.FuncAnimation(fig, update, G.shape[0]//interv, fargs=[line1,line2,line3],
                              interval=1, blit=True, save_count=50)

Writer = animation.writers['pillow']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=18000)

ani.save('./gyroscope.gif', writer=writer, dpi = 1000)

