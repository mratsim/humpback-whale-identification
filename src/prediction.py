# Copyright (c) 2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#              Prediction and final output
#
# ############################################################


import torch
import logging
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import os
from src.instrumentation import logspeed

## Get the same logger from main"
logger = logging.getLogger("humpback-whale")

##################################################
#### Prediction function
def predict(test_loader, model):
  model.eval()
  predictions = [] # Store predictions on CPU, we don't have enough GPU mem

  logger.info("Starting Prediction")
  with torch.no_grad():
    for batch_idx, data_target in enumerate(tqdm(test_loader)):
      data = data_target[0]["data"]

      pred = model(data)
      _, pred_encoded_labels = torch.topk(pred, 5)
      predictions.append(pred_encoded_labels.cpu())

  result = torch.cat(predictions, 0)
  logger.info("===> Predictions done. Here is a snippet (encoded labels)")
  logger.info(result)
  return result

@logspeed
def output(predictions, X_test, label_encoder, dir_path, run_name):
  result = pd.DataFrame({
    'Image': X_test['Image'],
    'Id': predictions
  })
  # The lambda and repeated inverse_transform calls are a recipe for slowness ...
  result['Id'] = result['Id'].apply(lambda whales: " ".join(label_encoder.inverse_transform(whales)))

  logger.info("===> Final predictions done. Here is a snippet")
  logger.info(result)

  result_path = os.path.join(dir_path, run_name + '-final-pred.csv')
  result.to_csv(result_path, index=False)
  logger.info(f"Final predictions saved to {result_path}")