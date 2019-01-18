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
from src.datafeed import *
from src.training import train, snapshot
from src.validation import validate
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
TRAIN_IMG_LIST = './preprocessing/fold0_train.txt'
VAL_IMG_LIST = './preprocessing/fold0_val.txt'
LABEL_ENCODER = './preprocessing/labelEncoder.pickle'

NUM_THREADS = 18
DATA_AUGMENT_GPU = 1      # GPU used for data augmentation.

EPOCHS = 30
BATCH_SIZE = 512          # This will be split onto all GPUs
VAL_BATCH_SIZE = 256      # Not sure why I have enough mem in forward+backward and not in validation
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
  train_pipe = SimplePipeline(
    img_dir=TRAIN_DIR,
    img_list_path=TRAIN_IMG_LIST,
    batch_size=BATCH_SIZE,
    crop_size=224,
    ch_mean = NORM_MEAN,
    ch_std = NORM_STD,
    num_threads = NUM_THREADS,
    device_id = DATA_AUGMENT_GPU,
    seed = 1337
    )
  train_pipe.build()

  val_pipe = ValPipeline(
    img_dir=TRAIN_DIR,
    img_list_path=VAL_IMG_LIST,
    batch_size=VAL_BATCH_SIZE,
    crop_size=224,
    ch_mean = NORM_MEAN,
    ch_std = NORM_STD,
    num_threads = NUM_THREADS,
    device_id = DATA_AUGMENT_GPU,
    seed = 1337
    )
  train_pipe.build()

  # Data loaders
  train_loader = DALIClassificationIterator(train_pipe, size = train_pipe.epoch_size("Datafeed"))
  val_loader = DALIClassificationIterator(val_pipe, size = train_pipe.epoch_size("Datafeed"))

  with open(LABEL_ENCODER, 'rb') as fh:
    le = pickle.load(fh)

  num_classes = le.classes_.size
  logger.info(f"Found {num_classes} unique classes to classify.")

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
      score, loss = validate(epoch, val_loader, model, criterion)

      # Save
      is_best = score > best_score
      best_score = max(score, best_score)
      snapshot(SAVE_DIR, RUN_NAME, is_best,{
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'best_score': best_score,
          'optimizer': optimizer.state_dict(),
          'val_loss': loss
      })

      # Cleanup
      train_loader.reset()
      val_loader.reset()

      end_epoch_timer = timer()
      logger.info("#### End epoch {}, elapsed time: {}".format(epoch, end_epoch_timer - epoch_timer))
      
  # ############################################################
  #
  #                     Cleanup
  #
  # ############################################################
  logging.shutdown()
  final_logfile = os.path.join('./outputs/', f'{str_timerun}--valid{val_score:.4f}.log')
  os.rename(tmp_logfile, final_logfile)

main()