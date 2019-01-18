from src.datafeed import SimplePipeline
from timeit import default_timer as timer

TRAIN_DIR = './input/train'
IMG_LIST = './preprocessing/input_dali.txt'

NUM_THREADS = 18
DATA_AUGMENT_GPU = 1

BATCH_SIZE = 256

# Normalization parameter (if pretrained, use the ones from ImageNet)
NORM_MEAN = [0.6073162, 0.5655911, 0.528621]   # ImgNet [0.485, 0.456, 0.406]
NORM_STD = [0.26327327, 0.2652084, 0.27765632] # ImgNet [0.229, 0.224, 0.225]

NUM_SAMPLES = 20

def speedtest():
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
    # warmup
    for i in range(5):
        pipe.run()
    # test
    start = timer()
    for i in range(NUM_SAMPLES):
        pipe.run()
    end = timer()
    print("Speed: {} imgs/s".format((NUM_SAMPLES * BATCH_SIZE)/(end - start)))


speedtest()