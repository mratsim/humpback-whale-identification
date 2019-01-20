# ############################################################
#
#           Learning rates from pending PyTorch PR
#
# ############################################################

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
import math

class CyclicLR(_LRScheduler):
    ## https://github.com/pytorch/pytorch/pull/2016
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode parameter is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_idx (int): The index of the last batch. Default: -1
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.step()
        >>>         train_batch(...)
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 optimizer,
                 base_lr=1e-3,
                 max_lr=6e-3,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 last_batch_idx=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        base_lrs = self._format_lr('base_lr', optimizer, base_lr)
        if last_batch_idx == -1:
            for base_lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = base_lr

        self.max_lrs = self._format_lr('max_lr', optimizer, max_lr)

        step_size_down = step_size_down or step_size_up
        self.total_size = float(step_size_up + step_size_down)
        self.step_ratio = float(step_size_up) / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        super(CyclicLR, self).__init__(optimizer, last_batch_idx)

    def _format_lr(self, name, optimizer, lr):
        """Return correctly formatted lr for each param group."""
        if isinstance(lr, (list, tuple)):
            if len(lr) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(lr)))
            return torch.tensor(lr)
        else:
            return lr * torch.ones(len(optimizer.param_groups))

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.
        """
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)
        return lrs

class CosineAnnealingRestartsLR(_LRScheduler):
    # https://github.com/pytorch/pytorch/pull/11104
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule with warm restarts, where :math:`\eta_{max}` is set to the
    initial learning rate, :math:`T_{cur}` is the number of epochs since the
    last restart and :math:`T_i` is the number of epochs in :math:`i`-th run
    (after performing :math:`i` restarts). If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2} \eta_{mult}^i (\eta_{max}-\eta_{min})
        (1 + \cos(\frac{T_{cur}}{T_i - 1}\pi))
        T_i = T T_{mult}^i
    Notice that because the schedule is defined recursively, the learning rate
    can be simultaneously modified outside this scheduler by other operators.
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that in the
    paper the :math:`i`-th run takes :math:`T_i + 1` epochs, while in this
    implementation it takes :math:`T_i` epochs only. This implementation
    also enables updating the range of learning rates by multiplicative factor
    :math:`\eta_{mult}` after each restart.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Length of the initial run (in number of epochs).
        eta_min (float): Minimum learning rate. Default: 0.
        T_mult (float): Multiplicative factor adjusting number of epochs in
            the next run that is applied after each restart. Default: 2.
        eta_mult (float): Multiplicative factor of decay in the range of
            learning rates that is applied after each restart. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T, eta_min=0, T_mult=2.0, eta_mult=1.0, last_epoch=-1):
        self.T = T
        self.eta_min = eta_min
        self.eta_mult = eta_mult

        if T_mult < 1:
            raise ValueError('T_mult should be >= 1.0.')
        self.T_mult = T_mult

        super(CosineAnnealingRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        if self.T_mult == 1:
            i_restarts = self.last_epoch // self.T
            last_restart = i_restarts * self.T
        else:
            # computation of the last restarting epoch is based on sum of geometric series:
            # last_restart = T * (1 + T_mult + T_mult ** 2 + ... + T_mult ** i_restarts)
            i_restarts = int(math.log(1 - self.last_epoch * (1 - self.T_mult) / self.T,
                                      self.T_mult))
            last_restart = int(self.T * (1 - self.T_mult ** i_restarts) / (1 - self.T_mult))

        if self.last_epoch == last_restart:
            T_i1 = self.T * self.T_mult ** (i_restarts - 1)  # T_{i-1}
            lr_update = self.eta_mult / self._decay(T_i1 - 1, T_i1)
        else:
            T_i = self.T * self.T_mult ** i_restarts
            t = self.last_epoch - last_restart
            lr_update = self._decay(t, T_i) / self._decay(t - 1, T_i)

        return [lr_update * (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    @staticmethod
    def _decay(t, T):
        """Cosine decay for step t in run of length T, where 0 <= t < T."""
        return 0.5 * (1 + math.cos(math.pi * t / T))