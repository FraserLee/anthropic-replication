import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

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
