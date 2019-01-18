# Copyright (c) 2018-2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#     Format train.csv and sample_submission.csv for DALI
#
# ############################################################

# DALI the Nvidia GPU data loader has an almost undocumented option
# called file_list for the FileReader operation
# that can take a list of files + label

# from nvidia.dali.ops import FileReader
# help(FileReader)

# class FileReader(builtins.object)
#  |  FileReader(**kwargs)
#  |
#  |  This is 'CPU' operator
#  |
#  |  Read (Image, label) pairs from a directory
#  |
#  |  Parameters
#  |  ----------
#  |  `file_root` : str
#  |                Path to a directory containing data files.
#  |  `file_list` : str, optional, default = ''
#  |                Path to the file with a list of pairs ``file label``
#  |                (leave empty to traverse the `file_root` directory to obtain files and labels)
#  |  `initial_fill` : int, optional, default = 1024
#  |                   Size of the buffer used for shuffling.
#  |  `num_shards` : int, optional, default = 1
#  |                 Partition the data into this many parts (used for multiGPU training).
#  |  `random_shuffle` : bool, optional, default = False
#  |                     Whether to randomly shuffle data.
#  |  `shard_id` : int, optional, default = 0
#  |               Id of the part to read.
#  |  `tensor_init_bytes` : int, optional, default = 1048576
#  |                        Hint for how much memory to allocate per image.

# and https://github.com/NVIDIA/DALI/blob/v0.7.0-dev/docs/examples/getting%20started.ipynb

# So we will create a list of labels and store a pre-processed file
# in the format expected by DALI

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

print("Preprocessing input image paths and labels")
# Load
df = pd.read_csv('./input/train.csv')
print(df.head())

# Encode labels in 0 ..< number of whales
le = LabelEncoder()
labels = le.fit_transform(df['Id'])
print(labels[0:5])

# Format input for DALI
df['Id'] = labels
df.to_csv('./preprocessing/input_dali.txt', sep=' ', index=False, header=False)

# Save the label encoder
with open('labelEncoder.pickle', 'wb') as fh:
  pickle.dump(le, fh)

del df
del le

# For test we don't have labels but DALI expects 2-level of nesting like "base_dir/label/img.jpg"
# so we will put dummy label instead
print("\nPreprocessing test image paths and adding a dummy label")
df = pd.read_csv('./input/sample_submission.csv')
print(df.head())

df['Id'] = -1
df.to_csv('./preprocessing/test_dali.txt', sep=' ', index=False, header=False)