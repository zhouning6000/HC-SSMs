import h5py
import numpy as np

# 打开 .h5 文件
with h5py.File('/root/Model_Phy/data/SEVIR_IR069_STORMEVENTS_2018_0101_0630.h5','r') as file:    # 假设我们想读取一个数据集叫 'IR069'
    data = h5py.File('/root/Model_Phy/data/SEVIR_IR069_STORMEVENTS_2018_0101_0630.h5','r')['ir069'][:]

# 将数据保存为 .npy 文件
np.save('/root/Model_Phy/data/SEVIR_IR069_STORMEVENTS_2018_0101_0630.npy', data)
