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


# Generate dummy data
input_dim = 100
hidden_dim = 500
num_samples = 1000
dummy_data = torch.randn(num_samples, input_dim)

# Initialize the autoencoder
autoencoder = Autoencoder(input_dim, hidden_dim)

# Loss function
criterion = nn.MSELoss()
l1_lambda = 0.001

def loss_function(recon_x, x, f):
    MSE = criterion(recon_x, x)
    L1 = l1_lambda * f.abs().sum()
    return MSE + L1

# Optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for data in dummy_data:
        data = data.view(1, -1)

        optimizer.zero_grad()
        reconstructed = autoencoder(data)
        loss = loss_function(reconstructed, data, autoencoder.encoder[0].weight)

        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete")
