import torch
import numpy as np


# 'cuda' device for supported nvidia GPU, 'mps' for Apple silicon (M1-M3)
device = torch.device(
    'cuda:7' if torch.cuda.is_available() else 'mps'\
        if torch.backends.mps.is_available() else 'cpu')


def from_numpy(
        x: np.ndarray,
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    # raise NotImplementedError('Convert numpy array to torch tensor here and send to device')
    return torch.from_numpy(x).to(device, dtype=dtype)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    # raise NotImplementedError('Convert torch tensor to numpy array here')
    # HINT: if using GPU, move the tensor to CPU before converting to numpy
    return x.cpu().detach().numpy() if x.is_cuda else x.detach().numpy()