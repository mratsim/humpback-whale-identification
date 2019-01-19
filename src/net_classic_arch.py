# Copyright (c) 2019 Mamy André-Ratsimbazafy
# All rights reserved

# ############################################################
#
#                    Classic NN Architecture
#
# ############################################################

import torch
from torch import nn
from torchvision.models import *

def scan_ft_extract_params(model, classifier_layer):
  list_generators = []
  for layer in model.children():
    if layer != classifier_layer:
      list_generators.append(layer.parameters())
  
  return (tensor for layer in list_generators for tensor in layer)

def initialize_model(
    model_family,
    model_name,
    num_classes,
    frozen_weights=False,
    use_pretrained=False,
    data_parallel=False,
    weights=None):
  # Initialize a model
  # Also returns "feat_extractor" and "classifier" parameters
  # that can be passed to an optimizer to create parameter groups with different learning rate
  model = globals()[model_name](pretrained=use_pretrained)
  
  if frozen_weights:
    for param in model.parameters():
      param.requires_grad = False

  classifier_field = ''

  if model_family == "resnet":
      num_ftrs = model.fc.in_features
      model.fc = nn.Linear(num_ftrs, num_classes)
      classifier_field = 'fc'

  elif model_family == "alexnet":
      num_ftrs = model.classifier[6].in_features
      model.classifier[6] = nn.Linear(num_ftrs,num_classes)
      classifier_field = 'classifier'

  elif model_family == "vgg":
      num_ftrs = model.classifier[6].in_features
      model.classifier[6] = nn.Linear(num_ftrs,num_classes)
      classifier_field = 'classifier'


  elif model_family == "squeezenet":
      model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
      classifier_field = 'classifier'


  elif model_family == "densenet":
      num_ftrs = model.classifier.in_features
      model.classifier = nn.Linear(num_ftrs, num_classes)
      classifier_field = 'classifier'

  elif model_family == "inception":
      # Handle the auxilary net
      num_ftrs = model.AuxLogits.fc.in_features
      model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
      # Handle the primary net
      num_ftrs = model.fc.in_features
      model.fc = nn.Linear(num_ftrs,num_classes)

      classifier_field = 'fc'


  else:
      print("Invalid model family, exiting...")
      exit()

  if weights:
    model.load_state_dict(weights)

  if data_parallel:
    model = nn.DataParallel(model).cuda()
    classifier = getattr(model.module, classifier_field).parameters()
    feat_extractor = scan_ft_extract_params(model.module, getattr(model.module, classifier_field))
  else:
    classifier = getattr(model, classifier_field).parameters()
    feat_extractor = scan_ft_extract_params(model, getattr(model, classifier_field))
  

  return model, feat_extractor, classifier

if __name__ == "__main__":
  print(globals())
  print(initialize_model('resnet', 'resnet18', 5005))