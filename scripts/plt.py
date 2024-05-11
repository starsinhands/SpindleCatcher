import numpy as np 
import matplotlib.pyplot as plt
filedir = r"../input/sleepedf.npy"
data = np.load(filedir,allow_pickle=True)
data = data.tolist()
channels = data.keys()
eegs = list(data.values())
resample_rate = 100.0
fig, axs = plt.subplots(nrows=len(channels), sharex=True)
for ax, channel, x in zip(axs, channels, eegs):
    t = np.arange(x.shape[0]) / resample_rate
    ax.plot(t, x, 'k-')
    ax.set_title(channel)
    ax.legend()
fig.show()
plt.show()
