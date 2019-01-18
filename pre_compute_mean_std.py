# Copyright (c) 2018-2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#                Compute the dataset per channel
#                  mean and standard deviation
#
# ############################################################

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Note, using the Welford method with running mean and variance is slow.
# But since it has to be done only once it might be worth it

# Image sizes go from 5959x695 to 77x30. Numpy needs the same resolution for all.
RESOLUTION = 224 # completely arbitrary :P.

if __name__ == "__main__":
    data = []
    df_train = pd.read_csv('./input/train.csv')

    for file in tqdm(df_train['Image'], miniters=256):
        img = cv2.imread(f'./input/train/{file}')
        data.append(cv2.resize(img,(RESOLUTION,RESOLUTION)))

    data = np.array(data, np.float32) / 255 # Must use float32 at least otherwise we get over float16 limits
    print("Shape: ", data.shape)

    means = []
    stdevs = []
    for i in range(3):
        # Data in format NHWC
        pixels = data[:,:,:,i].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))

# Shape:  (25361, 224, 224, 3)
# means: [0.6073162, 0.5655911, 0.528621]
# stdevs: [0.26327327, 0.2652084, 0.27765632]
# transforms.Normalize(mean = [0.6073162, 0.5655911, 0.528621], std = [0.26327327, 0.2652084, 0.27765632])