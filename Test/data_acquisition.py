#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().system('pip install pyedflib')

get_ipython().system('pip install mne')

import mne

raw = mne.io.read_raw_edf("1 (1).edf", preload=True)
data = raw.get_data()
channel_names = raw.ch_names
df = pd.DataFrame(data.T, columns=channel_names)
print(df.head())
df.dropna()   


# In[ ]:


class DataAcquisition:
    def __init__(self, TIME_STAMP_s, T8):
        self.TIME_STAMP_s=TIME_STAMP_s
        self.T8=T8
    def plot_histogram(self, figsize=(15,6)):
        plt.figure(figsize=figsize)
        bins = int(np.sqrt(len(self.T8)))
        plt.hist(self.T8, bins=bins, color='skyblue', edgecolor='black')
        plt.title('Histogram of T8 EEG Signal Amplitude')
        plt.xlabel('Amplitude')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.show()

    def plot_eeg_signal(self, figsize=(15, 6)):
        """
        Plot the T8 EEG signal over time.
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 6)
            Figure size as (width, height) in inches
        """
        plt.figure(figsize=figsize)
        plt.plot(self.TIME_STAMP_s, self.T8, linewidth=1)
        plt.title('EEG Signal from T8 Electrode')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (μV)')
        plt.grid(True)
        plt.show()


# In[ ]:


import matplotlib.pyplot as plt
df.plot(x='TIME_STAMP_s', y='T8', figsize=(15,6))
plt.title('T8_Horror')
plt.xlabel('time.sec')
plt.ylabel('T8')
plt.show


# In[ ]:




