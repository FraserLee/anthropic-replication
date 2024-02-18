import torch
from torch import nn
import torch.optim as optim


"""
Notes:
- one hidden layer MLP
- trained as an autoencoder using the input weights as an encoder and output weights as the decoder
- hidden layer is much wider than the inputs and applies a ReLU non-linearity
- Pytorch Kaiming Uniform initialization
"""

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize weights using Kaiming Uniform initialization
        self._initialize_weights()

    def forward(self, x):
        # Pre-encoder bias adjustment (decoder bias subtracted from input)
        x = x - self.decoder.bias
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def _initialize_weights(self):
        # Initialize encoder and decoder weights using Kaiming Uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
