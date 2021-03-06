# Copyright (c) 2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#                     Training routines
#
# ############################################################

import torch
import os
import logging
from src.validation import validate
from src.instrumentation import logspeed
from timeit import default_timer as timer

## Get the same logger from main"
logger = logging.getLogger("humpback-whale")

def train_at_epoch(epoch, train_loader, model, loss_func, optimizer, lr_scheduler, batch_size, report_freq):
    model.train() # training mode (build graph)
    
    for batch_idx, data_target in enumerate(train_loader):
        data = data_target[0]["data"]
        target = data_target[0]["label"].squeeze().type(torch.cuda.LongTensor)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if lr_scheduler:
          lr_scheduler.step()
        if batch_idx % report_freq == 0:
            logger.info('Train Epoch: {:03d} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, train_loader._size,
                100 * batch_idx * batch_size / train_loader._size, loss.item()))

def snapshot(dir_path, run_name, is_best, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '.pth')
    if is_best:
        torch.save(state, snapshot_file)
        logger.info(f"Snapshot saved to {snapshot_file}")

@logspeed
def train(
    model, train_loader,
    criterion, optimizer,
    batch_size, epochs, report_freq,
    snapshot_dir, run_name,
    data_parallel,
    evaluate, lr_scheduler=None, val_loader=None):
  if evaluate:
    best_score = 0.
  else:
    best_score = None
  for epoch in range(epochs):
      epoch_timer = timer()

      # Train and validate
      train_at_epoch(epoch, train_loader, model, criterion, optimizer, lr_scheduler, batch_size, report_freq)
      train_loader.reset()
      # TODO distinguish between lr_scheduler applied at each epochs and those applied at each iteration
      if evaluate:
        # Validate
        score, loss = validate(epoch, val_loader, model, criterion)
        val_loader.reset()

        # Save
        is_best = score > best_score
        best_score = max(score, best_score)
        snapshot(snapshot_dir, run_name, is_best,{
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if data_parallel else model.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
            'val_loss': loss
        })
      else:
        snapshot(snapshot_dir, run_name, True,{
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if data_parallel else model.state_dict(),
            'optimizer': optimizer.state_dict()
        })

      end_epoch_timer = timer()
      logger.info("#### End epoch {}, elapsed time: {}".format(epoch, end_epoch_timer - epoch_timer))

  return os.path.join(snapshot_dir, run_name + '.pth'), best_score
