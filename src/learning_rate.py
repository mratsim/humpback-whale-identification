# Copyright (c) 2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#                Learning rate scheduler
#
# ############################################################

# def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
#     """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
#     lr = init_lr * (0.1**(epoch // lr_decay_epoch))

#     if epoch % lr_decay_epoch == 0:
#         logger.info('LR is set to {}'.format(lr))

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

#     return optimizer