# Copyright (c) 2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#  Efficient data loading and augmentation using GPU routines
#
# ############################################################

import torch

# Nvidia DALI, GPU data loading and augmentation
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class SimplePipeline(Pipeline):
  def __init__(self, img_dir, img_list_path, batch_size, crop_size, ch_mean, ch_std, num_threads, device_id, seed = 1234):
    super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = seed)
    self.input = ops.FileReader(
      file_root = img_dir, file_list = img_list_path,
      random_shuffle = True, initial_fill = 2500 # This seems to use reservoir sampling
    )
    # On my workstation CPU is faster (4.5kimg/s vs 3kimg/s)
    # The squeezenet baseline pipelin takes 50s instead of 53s to run 3 epochs
    # I assume this is due to:
    #   1. I can use bigger batches if I don't need to keep a buffer for the image decoding
    #   2. OpenCV (the HostDecoder) was compiled with full Intel MKL + IPP + AVX512 optimisations.
    # self.decode = ops.nvJPEGDecoder(
    #   device="mixed",
    #   output_type=types.RGB,
    #   use_batched_decode= True
    #   )
    self.decode = ops.HostDecoder()
    self.rrc = ops.RandomResizedCrop(
      device = "cpu",
      size = crop_size,
      interp_type = types.INTERP_CUBIC
    )
    self.np = ops.NormalizePermute(
      height = crop_size,
      width = crop_size,
      device = 'cpu',
      mean = ch_mean,
      std = ch_std,
      image_type = types.RGB,
      output_dtype = types.FLOAT # Float16?
    )

  def define_graph(self):
    jpegs, labels = self.input(name = "Datafeed")
    images = self.decode(jpegs)
    images = self.rrc(images)
    images = self.np(images)
    return images, labels

class ValPipeline(Pipeline):
  def __init__(self, img_dir, img_list_path, batch_size, crop_size, ch_mean, ch_std, num_threads, device_id, seed = 1234):
    super(ValPipeline, self).__init__(batch_size, num_threads, device_id, seed = seed)
    self.input = ops.FileReader(
      file_root = img_dir, file_list = img_list_path,
      random_shuffle = False, initial_fill = 2500
    )
    # self.decode = ops.nvJPEGDecoder(
    #   device="mixed",
    #   output_type=types.RGB,
    #   use_batched_decode= True
    #   )
    self.decode = ops.HostDecoder()
    self.res = ops.Resize(device="cpu", resize_shorter=crop_size)
    self.cmnp = ops.CropMirrorNormalize(device="cpu",
                                        output_dtype=types.FLOAT,
                                        output_layout=types.NCHW,
                                        crop=(crop_size, crop_size),
                                        image_type=types.RGB,
                                        mean=ch_mean,
                                        std=ch_std)

  def define_graph(self):
    jpegs, labels = self.input(name = "Datafeed")
    images = self.decode(jpegs)
    images = self.res(images)
    images = self.cmnp(images)
    return images, labels

if __name__ == "__main__":
  # help(ops.nvJPEGDecoder)
  help(ops)