# ############################################################
#
#                           Imports
#
# ############################################################

import torch
from torch import nn, optim
from torch.nn import DataParallel
import torch.nn.functional as F

# Nvidia Apex, fp16 and mixed precision training
from apex.fp16_utils import *

# Nvidia DALI, GPU Data Augmentation Library
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

# Utilities
import random
import logging
import time
from timeit import default_timer as timer
import os
import pickle

# local import
from src.instrumentation import setup_logs
from src.datafeed import SimplePipeline
from src.training import train
from src.net_squeezenet import *

# ############################################################
#
#                     Environment variables
#
# ############################################################

# Setting random seeds for reproducibility.
# Parallel sum reductions are non-deterministic due to the non-associativity of floating points
# Note that matrix multiplication and convolution are doing sum reductions ¯\_(ツ)_/¯
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
# np.random.seed(1337)
random.seed(1337)

SAVE_DIR = './snapshots' # Path for save intermediate and final weights of models
TRAIN_DIR = './input/train'
IMG_LIST = './preprocessing/input_dali.txt'
LABEL_ENCODER = './preprocessing/labelEncoder.pickle'

NUM_THREADS = 18
DATA_AUGMENT_GPU = 1      # GPU used for data augmentation.

EPOCHS = 3
BATCH_SIZE = 512          # This will be split onto all GPUs
REPORT_EVERY_N_BATCH = 5

# Normalization parameter (if pretrained, use the ones from ImageNet)
NORM_MEAN = [0.6073162, 0.5655911, 0.528621]   # ImgNet [0.485, 0.456, 0.406]
NORM_STD = [0.26327327, 0.2652084, 0.27765632] # ImgNet [0.229, 0.224, 0.225]

RUN_NAME = time.strftime("%Y-%m-%d_%H%M-") + "BASELINE"
TMP_LOGFILE = os.path.join('./outputs/', f'{RUN_NAME}--run-in-progress.log')

def main():
  global_timer = timer()
  logger = setup_logs(TMP_LOGFILE)
  
  # ############################################################
  #
  #                     Processing pipeline
  #
  # ############################################################
  pipe = SimplePipeline(
    img_dir=TRAIN_DIR,
    img_list_path=IMG_LIST,
    batch_size=BATCH_SIZE,
    crop_size=224,
    ch_mean = NORM_MEAN,
    ch_std = NORM_STD,
    num_threads = NUM_THREADS,
    device_id = DATA_AUGMENT_GPU,
    seed = 1337
    )
  pipe.build()

  # Data loader
  train_loader = DALIClassificationIterator(pipe, size = pipe.epoch_size("Datafeed"))

  with open(LABEL_ENCODER, 'rb') as fh:
    le = pickle.load(fh)

  num_classes = le.classes_.size
  logging.info(f"Found {num_classes} to classify.")

  # I choose SqueezeNet for speed for the baseline
  model = SqueezeNetv1_1(le.classes_.size)
  model = DataParallel(model).cuda()
  optimizer = optim.Adam(model.parameters())
  criterion = nn.CrossEntropyLoss()

  # ############################################################
  #
  #                     Training
  #
  # ############################################################
  best_score = 0.
  for epoch in range(EPOCHS):
      epoch_timer = timer()

      # Train and validate
      train(epoch, train_loader, model, criterion, optimizer, BATCH_SIZE, REPORT_EVERY_N_BATCH)

      # Reset data loader
      train_loader.reset()

  # ############################################################
  #
  #                     Cleanup
  #
  # ############################################################
  logging.shutdown()
  # final_logfile = os.path.join('./outputs/', f'{str_timerun}--valid{val_score:.4f}.log')
  # os.rename(tmp_logfile, final_logfile)

main()