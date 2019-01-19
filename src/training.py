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

def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        logger.info('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_at_epoch(epoch, train_loader, model, loss_func, optimizer, batch_size, report_freq):
    model.train() # training mode (build graph)
    optimizer = lr_scheduler(optimizer, epoch)
    
    for batch_idx, data_target in enumerate(train_loader):
        data = data_target[0]["data"]
        target = data_target[0]["label"].squeeze().type(torch.cuda.LongTensor)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % report_freq == 0:
            logger.info('Train Epoch: {:03d} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, train_loader._size,
                100 * batch_idx * batch_size / train_loader._size, loss.item()))

def snapshot(dir_path, run_name, is_best, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
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
    evaluate, val_loader=None):
  best_score = 0.
  for epoch in range(epochs):
      epoch_timer = timer()

      # Train and validate
      train_at_epoch(epoch, train_loader, model, criterion, optimizer, batch_size, report_freq)
      train_loader.reset()

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

  return os.path.join(snapshot_dir, run_name + '-model_best.pth')