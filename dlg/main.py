# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

# Selecting cpu vs gpu
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

# Download dataset and define transformations
dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# Define a parser to use images from the data set and custom images
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default="25", help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str,default="", help='the path to customized image.')
args = parser.parse_args()

# Define which image to use for the ground truth
img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)
if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)

# Compute properties of the ground truth image
gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

# Plot the ground truth image
plt.imshow(tt(gt_data[0].cpu()))
plt.title("Ground truth image")
print("GT label is %d." % gt_label.item(), "\nOnehot label is %d." % torch.argmax(gt_onehot_label, dim=-1).item())

# Import and define the network
from models.vision import LeNet, weights_init
net = LeNet().to(device)

# Set a seed
torch.manual_seed(1234)

# Initialise the network
net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute original gradient 
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())
original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# add noise (differential privacy) to the gradient?
# Laplacian noise
noise = 0.002
for idx in range(len(original_dy_dx)):
    original_dy_dx[idx] += np.random.laplace(0,noise,original_dy_dx[idx].size())

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

# Define an optimizer
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

# Store intermediary results
history = []

# Reconstruct the image in 300 steps
for iters in range(300):
    # TODO: What is this doing exactly?
    def closure():
        # Set all gradients to zero before next iteration
        optimizer.zero_grad()

        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    # Gradient step; iterate the network 1 step
    optimizer.step(closure)

    # Print intemediary results every 10 iterations
    if iters % 10 == 0: 
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

# Plot the final results
plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')
plt.show()
