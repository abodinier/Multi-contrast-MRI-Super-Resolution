import os
import re
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from math import log10
import torch.optim as optim
from model.edsr import EDSR
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from utils.dataset import load_img, convert2pilImage

parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('--model_path', type=str, metavar='N',
                    help='pre-trained model path')
parser.add_argument('--input_channels',
                type=lambda s: re.split(' |, ', s),
                required=False,
                help='input channels (default : t1w + t1gd + t2w + flair)',
                default=["t1w", "t1gd", "t2w", "flair"])
parser.add_argument('--target_channels',
                type=lambda s: re.split(' |, ', s),
                required=False,
                help='target channels (default : t1w + t1gd + t2w + flair)',
                default=["t1w", "t1gd", "t2w", "flair"])
parser.add_argument('--output_dir', type = str, default=os.getcwd()+'/', help='default = ./output')
parser.add_argument('--image_path', type = str)
parser.add_argument('--verbose', type = bool, default="false", help='default = false')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocessing(data, input_channels):
    imgs = [(convert2pilImage(data[x])) for x in input_channels]
    imgs = [ToTensor()(x) for x in imgs]
    (_, h, w) = imgs[0].shape
    img_tensor = torch.stack(imgs, dim=1).view(1, len(input_channels), h, w)
    return img_tensor

def calc_psnr(pred, target):
    predd = ToTensor()(pred)
    targett = ToTensor()(target)
    print("pred: ", predd.size(), "target: ", targett.size())
    mse = nn.MSELoss()(predd, targett)
    return 10*log10(1/mse)

data = load_img(args.image_path) 

model = torch.load(args.model_path).to(device)
inp = preprocessing(data, args.input_channels)
model.load_state_dict(torch.load(args.model_path, map_location=device))

output = model(inp)

output_img = output.cpu()[0][0].detach().numpy()
norm = 255.
output_img *= norm
output_img.clip(0, 255)
output_img = Image.fromarray(output_img, mode='F').convert("L")
output_img.save("SR.jpg", quality=100)