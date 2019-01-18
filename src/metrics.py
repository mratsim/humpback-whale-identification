# Copyright (c) 2019 Mamy André-Ratsimbazafy
# All rights reserved

import torch

def ap_at_5(predictions, ground_truth):
  ## Average Precision of a batch of predictions
  ## See: https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric
  ## Input:
  ##   - A tensor of predictions as float probabilities of shape [Batch_size, Nb_of_Whale_IDs]
  ##   - A tensor of integer truth labels of shape [Batch_size]
  ##
  ## ⚠ Truth labels must be encoded with an integer in range 0 ..< Nb_of_Whale_IDs

  _, pred_encoded_labels = torch.topk(predictions, 5) # returns (predictions, labels) in descending order

  ap5 = torch.zeros(ground_truth.size(), device = predictions.device) # Can be preallocated and reused if bottleneck
  for rank in range(5):
    # ap5 = if truth: 1/(rank+1) else: 0
    torch.add(ap5, 1./(rank+1), (pred_encoded_labels[:, rank]==ground_truth).float(), out=ap5) # ap5 += 1/rank * (pred[rank] == truth)
  return ap5

def map_at_5(predictions, ground_truth):
  ## Mean Average Precision of a batch of predictions
  return torch.mean(ap_at_5(predictions, ground_truth))

# Sanity checks
if __name__ == "__main__":
  import torch.nn.functional as F
  torch.manual_seed(1337)

  for n in range(5):
    print(f"\n########## Iteration {n}")
    # To check our implementation we will pick 3x10 probabilities (batch of 3 images with 10 classes)
    preds = torch.rand(3, 10) # Uniform random number
    preds = F.softmax(preds, dim=1)  # Transform those in a probability distribution: each rows must sum to 1

    # Ground Truth
    truth = torch.tensor([3, 6, 9])

    print(preds)
    print("Top-5")
    print(torch.topk(preds, 5))
    print("mAP@5")
    print(map_at_5(preds, truth))