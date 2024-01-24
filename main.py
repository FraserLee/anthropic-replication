import torch

if torch.cuda.is_available():
    print(torch.cuda.current_device())
else:
    print("torch run on cpu")

