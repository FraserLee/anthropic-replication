import torch
from torch import nn


"""
Notes:
- one hidden layer MLP
- trained as an autoencoder using the input weights as an encoder and output weights as the decoder
- hidden layer is much wider than the inputs and applies a ReLU non-linearity
- Pytorch Kaiming Uniform initialization
"""

class Autoencoder(nn.Module):
    pass
