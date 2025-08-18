import os
import numpy as np
path="/root/autodl-tmp/EarthFarseer-main/Dataset/SEVIR_IR069_STORMEVENTS.npy"
data = np.load(path)
data_1=data+32768
data_2=data_1*0.0000294
np.save("/root/autodl-tmp/EarthFarseer-main/Dataset/SEVIR_IR069_STORMEVENTS_bk.npy", data_2)

