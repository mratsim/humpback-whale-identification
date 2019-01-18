# Copyright (c) 2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#                    SqueezeNet Arch
#
# ############################################################

import torch
from torch import nn
from torchvision import models

class SqueezeNetv1_1(nn.Module):
  def __init__(self, num_classes):
    super(SqueezeNetv1_1, self).__init__()
    original_model = models.squeezenet1_1(pretrained=False)

    self.features = original_model.features
    self.classifier = original_model.classifier
    # For SqueezeNet 1.0
    # (classifier): Sequential(
    #     (0): Dropout(p=0.5)
    #     (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
    #     (2): ReLU(inplace)
    #     (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
    # )
    self.classifier[1] = nn.Conv2d(self.classifier[1].in_channels, num_classes, kernel_size=(1,1), stride=(1,1))

  def forward(self, x):
      result = self.features(x)
      result = self.classifier(result)
      result = result.view(result.size(0), -1) # Flatten
      return result


if __name__ == "__main__":
  print(SqueezeNetv1_1(5005))