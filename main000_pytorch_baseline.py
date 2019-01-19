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

# Output
import pandas as pd

# local import
from src.instrumentation import setup_logs, logspeed
from src.datafeed import *
from src.net_classic_arch import *
from src.training import train
from src.prediction import predict, output

# Command-line interface
from argparse import ArgumentParser

# ############################################################
#
#                           Parser
#
# ############################################################

def parse_args():
    parser = ArgumentParser(description="Train/Validate on fold."
                                        " Train on full dataset."
                                        " Predict using an existing weight.")
    parser.add_argument('--fulldata', '-fd', action='store_true', default=False,
                         help='Train on the full dataset instead of fold 0')
    parser.add_argument('--predictonly', '-po', type=str, default=None,
                         help='Predict only, using weights at the specified path.')
    parser.add_argument('--dataparallel', '-dp', action='store', default=True,
                         help='Use DataParallel')       
    return parser.parse_args()

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

WEIGHTS_DIR = './snapshots' # Path for save intermediate and final weights of models
TRAIN_DIR = './input/train'
TRAIN_FULL_IMG_LIST = './preprocessing/full_input.txt'
TRAIN_FOLD_IMG_LIST = './preprocessing/fold0_train.txt'
VAL_IMG_LIST = './preprocessing/fold0_val.txt'
LABEL_ENCODER_PATH = './preprocessing/labelEncoder.pickle'

TEST_DIR = './input/test'
TEST_IMG_LIST = './preprocessing/test_data.txt'
SUBMISSION_FILE = './input/sample_submission.csv'
OUT_DIR = './outputs'

NUM_THREADS = 18

EPOCHS = 30
BATCH_SIZE = 96          # This will be split onto all GPUs
VAL_BATCH_SIZE = 256     # We can use large batches when weights are frozen
REPORT_EVERY_N_BATCH = 5

PRETRAINED = True
UNFROZE_AT_EPOCH = 3
BATCH_FROZEN = 256       # We can use large batches when weights are frozen

if PRETRAINED:
  # ImgNet normalization
  NORM_MEAN = [0.485, 0.456, 0.406]
  NORM_STD = [0.229, 0.224, 0.225]
else:
  # Dataset normalization parameter
  NORM_MEAN = [0.6073162, 0.5655911, 0.528621]   # ImgNet [0.485, 0.456, 0.406]
  NORM_STD = [0.26327327, 0.2652084, 0.27765632] # ImgNet [0.229, 0.224, 0.225]

with open(LABEL_ENCODER_PATH, 'rb') as fh:
  LABEL_ENCODER = pickle.load(fh)

model = initialize_model(
  model_family = 'resnet',
  model_name = 'resnet101',
  num_classes = LABEL_ENCODER.classes_.size,
  frozen_weights = PRETRAINED,
  use_pretrained = PRETRAINED
  )
LOG_SUFFIX = "resnet101-baseline"

# GPU data augmentation
# Note that it's probably better to do augmentation on CPU for compute intensive models
# So that you can maximize the batch size and training on GPU.
DATA_AUGMENT_USE_GPU = True
DATA_AUGMENT_GPU_DEVICE = 1

@logspeed
def main_train(model, args, run_name, logger):
  # ############################################################
  #
  #                     Processing pipeline
  #
  # ############################################################
  train_pipe = SimplePipeline(
    img_dir=TRAIN_DIR,
    img_list_path= TRAIN_FULL_IMG_LIST if args.fulldata else TRAIN_FOLD_IMG_LIST,
    batch_size=BATCH_SIZE,
    crop_size=224,
    ch_mean = NORM_MEAN,
    ch_std = NORM_STD,
    num_threads = NUM_THREADS,
    use_gpu = DATA_AUGMENT_USE_GPU,
    gpu_id = DATA_AUGMENT_GPU_DEVICE,
    seed = 1337
    )
  train_pipe.build()
  train_loader = DALIClassificationIterator(train_pipe, size = train_pipe.epoch_size("Datafeed"))
  
  if not args.fulldata:
    val_pipe = ValPipeline(
      img_dir=TRAIN_DIR,
      img_list_path=VAL_IMG_LIST,
      batch_size=VAL_BATCH_SIZE,
      crop_size=224,
      ch_mean = NORM_MEAN,
      ch_std = NORM_STD,
      num_threads = NUM_THREADS,
      use_gpu = DATA_AUGMENT_USE_GPU,
      gpu_id = DATA_AUGMENT_GPU_DEVICE,
      seed = 1337
      )
    val_pipe.build()
    val_loader = DALIClassificationIterator(val_pipe, size = val_pipe.epoch_size("Datafeed"))

  num_classes = LABEL_ENCODER.classes_.size
  logger.info(f"Found {num_classes} unique classes to classify.")
  model_name = model.module.__class__.__name__ if args.dataparallel else model.__class__.__name__

  # optimizer = optim.Adam(model.parameters())
  optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)

  criterion = nn.CrossEntropyLoss()

  # If network is pretrained, we need to freeze feature layers
  # first so that the classifier adapt to their output range
  # after a couple epoch we can unfreeze to train everything
  if PRETRAINED:
    pretraining_pipe = SimplePipeline(
      img_dir=TRAIN_DIR,
      img_list_path= TRAIN_FULL_IMG_LIST if args.fulldata else TRAIN_FOLD_IMG_LIST,
      batch_size=BATCH_FROZEN,
      crop_size=224,
      ch_mean = NORM_MEAN,
      ch_std = NORM_STD,
      num_threads = NUM_THREADS,
      use_gpu = DATA_AUGMENT_USE_GPU,
      gpu_id = DATA_AUGMENT_GPU_DEVICE,
      seed = 1337
      )
    pretraining_pipe.build()
    pretrain_loader = DALIClassificationIterator(pretraining_pipe, size = pretraining_pipe.epoch_size("Datafeed"))

    logger.info(f"Pre-training with frozen feature extraction layers with batch size {BATCH_FROZEN} for {UNFROZE_AT_EPOCH} epochs")
    weights = train(
      model = model, train_loader = pretrain_loader,
      criterion = criterion, optimizer = optimizer,
      batch_size = BATCH_FROZEN, epochs = UNFROZE_AT_EPOCH,
      report_freq = REPORT_EVERY_N_BATCH,
      snapshot_dir = WEIGHTS_DIR,
      run_name = run_name,
      data_parallel = args.dataparallel,
      evaluate = not args.fulldata,
      val_loader = None if args.fulldata else val_loader,
    )
    logger.info(f"End pretraining, unfroze all weights.")

  logger.info(f"Training {model.module.__class__.__name__} with batch size {BATCH_SIZE} for {EPOCHS} epochs.")
  weights = train(
    model = model, train_loader = train_loader,
    criterion = criterion, optimizer = optimizer,
    batch_size = BATCH_SIZE, epochs = EPOCHS,
    report_freq = REPORT_EVERY_N_BATCH,
    snapshot_dir = WEIGHTS_DIR,
    run_name = run_name,
    data_parallel = args.dataparallel,
    evaluate = not args.fulldata,
    val_loader = None if args.fulldata else val_loader,
  )
  return weights

@logspeed
def main_predict(model):
  test_pipe = ValPipeline(
      img_dir=TEST_DIR,
      img_list_path=TEST_IMG_LIST,
      batch_size=VAL_BATCH_SIZE,
      crop_size=224,
      ch_mean = NORM_MEAN,
      ch_std = NORM_STD,
      num_threads = NUM_THREADS,
      use_gpu = DATA_AUGMENT_USE_GPU,
      gpu_id = DATA_AUGMENT_GPU_DEVICE,
      seed = 1337
      )
  test_pipe.build()
  test_loader = DALIClassificationIterator(test_pipe, size = test_pipe.epoch_size("Datafeed"))
  return predict(test_loader, model)

@logspeed
def main(model):
  args = parse_args()

  if args.dataparallel:
    model = DataParallel(model).cuda()
  else:
    model = model.cuda()

  model_weights_path = ''
  if args.predictonly:
    # Pretrained
    run_name = time.strftime("%Y-%m-%d_%H%M-") + LOG_SUFFIX + ("-fulldata" if args.fulldata else "")
    tmp_logfile = os.path.join(OUT_DIR, f'{run_name}--run-in-progress.log')

    logger = setup_logs(tmp_logfile)
    model_weights_path = args.predictonly

  else:
    # Training
    run_name = time.strftime("%Y-%m-%d_%H%M-") + f"{LOG_SUFFIX}"
    tmp_logfile = os.path.join(OUT_DIR, f'{run_name}--run-in-progress.log')

    logger = setup_logs(tmp_logfile)
    model_weights_path = main_train(model, args, run_name, logger)

  # Load model
  logger.info(f'===> loading model for prediction: {model_weights_path}')
  checkpoint = torch.load(model_weights_path)
  model.load_state_dict(checkpoint['state_dict'])

  # Predict
  pred = main_predict(model)

  # Output
  X_test = pd.read_csv(SUBMISSION_FILE)
  output(pred, X_test, LABEL_ENCODER, OUT_DIR, run_name)

  # ############################################################
  #
  #                     Cleanup
  #
  # ############################################################

  if evaluate:
    final_logfile = os.path.join('./outputs/', f'{run_name}.log')
    os.rename(TMP_LOGFILE, final_logfile)
  else:
    final_logfile = os.path.join('./outputs/', f'{run_name}--best_val_score-{best_score:.4f}.log')
    os.rename(TMP_LOGFILE, final_logfile)
  logger.info("   ===>  Finished all tasks!")

main(model)
logging.shutdown()