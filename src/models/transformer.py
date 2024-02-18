import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# TODO: SUPER MESSY AND SOMEWHAT BAD, REVIEW AND CLEAN UP, ESP SOS

"""
Notes:
- Adam optimizer
- residual stream dimension of 128
- inner MLP dimension of 512
- MLP activation is a ReLU

From paper:
```
Transformer Training and Preprocessing
The one-layer transformers we study are trained on the Pile. We chose to train these models on the Pile over Anthropic's internal dataset in order to make our experiments more reproducible. While we think many features are universal, others are very likely idiosyncratic to the dataset.

We train the transformers on 100 billion tokens using the Adam optimizer. We hypothesize that a very high number of training tokens may allow our model to learn cleaner representations in superposition. These transformers have a residual stream dimension of 128, and an inner MLP dimension of 512. The MLP activation is a ReLU.
```
"""

transformer = nn.Transformer(
    d_model=128,
    nhead=8,  # Standard choice
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=512,
    dropout=0.1,
    activation=F.relu,
    
    batch_first=True, # tmp
    # device=None,
)
# Note: Other parameters like batch_first, norm_first, bias, etc., are set to their default values as they are not specified in the paper.


class DummyDataset(Dataset):
    def __init__(self, num_samples, sequence_length, feature_size, num_classes):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.sos_token = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random data for source and target with correct batch dimensions
        src = torch.randn(self.sequence_length, self.feature_size)
        # Add an SOS token at the start of each target sequence
        tgt = torch.randint(0, self.num_classes, (self.sequence_length,))
        tgt = torch.cat([torch.tensor([self.sos_token]), tgt])
        return src, tgt


# Parameters for the dummy dataset
num_samples = 1000
sequence_length = 10
input_size = 128
num_classes = 128
batch_size = 32

# Initialize dataset and dataloader
dummy_dataset = DummyDataset(num_samples, sequence_length, input_size, num_classes)
train_loader = DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

# Optimizer and loss function
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Define the embedding layer for target, including the SOS token
tgt_embedding = nn.Embedding(num_classes + 1, input_size)  # +1 for SOS token

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for src, tgt in train_loader:
        optimizer.zero_grad()

        # Prepare the target tensor for 'teacher forcing'
        tgt_input = tgt[:, :-1]  # Ignore the last token for input
        tgt_expected = tgt[:, 1:].reshape(-1)  # Expected output (shifted by one position)

        # Embed the target tensor
        tgt_input_embedded = tgt_embedding(tgt_input)

        # Transpose the tensors to match Transformer's expected input shape
        src = src.transpose(0, 1)  # Shape: (sequence_length, batch_size, d_model)
        tgt_input_embedded = tgt_input_embedded.transpose(0, 1)  # Shape: (sequence_length, batch_size, d_model)

        # Forward pass
        output = transformer(src, tgt_input_embedded)
        output = output.transpose(0, 1).reshape(-1, num_classes)

        # Loss and backward pass
        loss = loss_fn(output, tgt_expected)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
