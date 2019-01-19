# Copyright (c) 2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#                  Validation of DL models
#
# ############################################################

import torch
import logging
import torch.nn.functional as F
from tqdm import tqdm

from src.metrics import map_at_5

## Get the same logger from main"
logger = logging.getLogger("humpback-whale")

##################################################
#### Validate function
def validate(epoch,valid_loader,model,loss_func):    
    model.eval() # Eval mode (no graph)
    total_loss = 0
    total_map5 = 0
    batch_count = 1
    
    logger.info("Starting Validation")
    with torch.no_grad():
      for batch_idx, data_target in enumerate(tqdm(valid_loader)):    
          data = data_target[0]["data"]
          target = data_target[0]["label"].squeeze().type(torch.cuda.LongTensor)
      
          pred = model(data)
          
          total_loss += loss_func(pred,target).item()
          total_map5 += map_at_5(pred, target).item()
          batch_count += 1
    
    avg_loss = total_loss / valid_loader._size
    map5 = total_map5 / batch_count
    
    logger.info("===> Validation - Avg. loss: {:.4f}\tMAP@5: {:.4f}".format(avg_loss,map5))
    return map5, avg_loss