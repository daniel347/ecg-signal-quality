import torch


def get_torch_device(enable_cuda=True):
    if torch.cuda.is_available() and enable_cuda:
        print("Using Cuda")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")
