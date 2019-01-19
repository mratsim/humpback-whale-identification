# Copyright (c) 2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#              Prediction and final output
#
# ############################################################

import numpy as np
import torch
import logging
import torch.nn.functional as F
import pandas as pd
import os
from src.instrumentation import logspeed

## Get the same logger from main"
logger = logging.getLogger("humpback-whale")

##################################################
#### Prediction function
def predict(test_loader, model, final_activation):
  model.eval()
  predictions = [] # Store predictions on CPU, we don't have enough GPU mem

  logger.info("Starting Prediction")
  with torch.no_grad():
    for batch_idx, data_target in enumerate(test_loader):
      data = data_target[0]["data"]

      pred = final_activation(model(data))
      predictions.append(pred.cpu())

  result = torch.cat(predictions, 0)
  logger.info("===> Predictions done. Here is a snippet (label probabilities)")
  logger.info(result)
  return result

@logspeed
def output(raw_predictions, X_test, label_encoder, dir_path, run_name):
  raw_pred_path = os.path.join(dir_path, run_name + '-raw-pred.csv')
  np.savetxt(raw_pred_path, raw_predictions,delimiter=";")
  logger.info("Raw predictions saved to {}".format(raw_pred_path))
  
  _, predictions = torch.topk(raw_predictions, 5)

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