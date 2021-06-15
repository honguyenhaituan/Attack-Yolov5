import torch
from torch import nn 
from math import log10

def cosine(img_1, img_2): 
    vector_1 = torch.flatten(img_1, start_dim=1).float()
    vector_2 = torch.flatten(img_2, start_dim=1).float()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(vector_1, vector_2)
    return torch.sum(output).item()

def psnr(img_1, img_2):
    criterion = nn.MSELoss()
    mse = criterion(img_1, img_2)
    return 10 * log10(1 / mse.item())

def ssim(img_1, img_2): 
    return None