import torch
import opt

device = torch.device('cuda:0')


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(opt.STD).to(device) + torch.Tensor(opt.MEAN).to(device)
    x = x.transpose(1, 3)
    return x
