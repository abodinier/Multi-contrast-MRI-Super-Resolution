# standard imports
import os
import re
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from math import log10
import torch.optim as optim
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from utils.dataset import DataLoader
from torchvision.transforms import ToTensor

# Own packages
from model.edsr import EDSR
from model.ssim import ssim
from utils.dataset import get_training_set, get_test_set
from utils.utils import calc_psnr, calc_ssim, show, save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('--batchSize', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--testBatchSize', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--nEpochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--cold_start', type=bool, default=False, metavar='CS',
                    help='if cold_start = True, train from scratch (default False)')
parser.add_argument('--upscale_factor', type=int, default=3, metavar='UF',
                    help='upscale_factor (default: 3)')
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
parser.add_argument('--n_resblocks', type=int, default=16, metavar='NRB',
                    help='n_resblocks (default: 16)')
parser.add_argument('--n_patch_features', type=int, default=256, metavar='NPF',
                    help='n_patch_features (default: 256)')
parser.add_argument('--modelName', type=str, default="Multi_contrast_SR", metavar='MCSR',
                    help='n_patch_features (default: 256)')
parser.add_argument('--data_dir', type=str, default=os.getcwd()+'/data', help='default = ./data')
parser.add_argument('--output_dir', type = str, default=os.getcwd()+'/', help='default = ./output')
parser.add_argument('--early_upsampling', type = bool, default=False, help='default = False')
parser.add_argument('--model_name', type = str, default="model", help='default = model')
parser.add_argument('--verbose', type = bool, default="false", help='default = false')
args = parser.parse_args()
    
# Training settings:
data_dir = args.data_dir
batchSize = args.batchSize
testBatchSize = args.testBatchSize
nEpochs = args.nEpochs
lr = args.lr
threads = 4
seed = 123
torch.manual_seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.cuda.device("cuda" if use_cuda else "cpu")
cold_start = args.cold_start
early_upsampling = args.early_upsampling

# Model :
upscale_factor = args.upscale_factor
input_channels = args.input_channels
target_channels = args.target_channels
patch_size = [240, 240]
n_patch_features = args.n_patch_features
n_resblocks = args.n_resblocks

# Paths :
training_path = os.path.join(data_dir, "train")
test_path = os.path.join(data_dir, "test")

training_folder = os.path.join("./", "trainings/", args.modelName)
model_weights = os.path.join("./", training_folder, "Pre_trained_model/")
stats_path = os.path.join("./", training_folder, "stats/")
images_path = os.path.join("./", training_folder, "/training_images/")
summary_path = os.path.join("./", training_folder, "/summary/")

paths = [training_folder, model_weights, stats_path, images_path, summary_path]
for path in paths:
    try:
        os.stat(path)
    except:
        os.makedirs(path) 

if args.verbose:
    print("input modalities = {}\ntarget modalities = {}".format(input_channels, target_channels))
    
if args.verbose:
    print('===> Loading datasets')
    
train_set = get_training_set(upscale_factor, input_channels, target_channels, training_path, patch_size=patch_size, early_upsampling=early_upsampling)
test_set = get_test_set(upscale_factor, input_channels, target_channels, test_path, patch_size=patch_size, early_upsampling=early_upsampling)

training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=testBatchSize, shuffle=False)

if args.verbose:
    print("{} training images / {} testing images".format(len(train_set), len(test_set)))
    print("===> dataset loaded !")
    print('===> Building model')
    

model = EDSR(upscale_factor, input_channels, target_channels, n_resblocks=n_resblocks, n_feats=n_patch_features, res_scale=.1, bn=None).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

if args.verbose:
    print(model.parameters())
    print(model)
    print("==> Model built")

def train(epoch):
    epoch_loss = 0
    epoch_loss_indiv = [0 for x in range(len(target_channels))]
    epoch_ssim_indiv = [0 for x in range(len(target_channels))]
    for iteration, batch in enumerate(training_data_loader, 1):
        inp, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        prediction = model(inp)

        loss = 0
        for x in range(len(target_channels)):
            loss_x = criterion(prediction[:, x, :, :], target[:, x, :, :])
            ssim_x = calc_ssim(prediction[:, x, :, :].view(len(batch[0]), 1, 240, 240), target[:, x, :, :].view(len(batch[0]), 1, 240, 240))
            epoch_loss_indiv[x] += loss_x.item()
            epoch_ssim_indiv[x] += ssim_x
            loss += loss_x

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    epochLoss = epoch_loss / len(training_data_loader)

    for x in range(len(target_channels)):
            epoch_loss_indiv[x] /= len(training_data_loader)
            epoch_ssim_indiv[x] /= len(training_data_loader)

    psnr_indiv = list(map(calc_psnr, epoch_loss_indiv))
    psnr = calc_psnr(epochLoss)
    print("psnr_indiv= ", psnr_indiv, "global_psnr= ", psnr)

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epochLoss))
    pred = prediction.cpu()[0]
    target = target.cpu()[0]

    for p, t, c_name in zip(pred, target, target_channels):
        show(epoch, p, t, criterion, images_path, args.model_name, title="train_"+c_name)
    return psnr, psnr_indiv, epoch_ssim_indiv


def test(epoch):
    avg_psnr = 0
    epoch_loss_indiv = [0 for x in range(len(target_channels))]
    epoch_ssim_indiv = [0 for x in range(len(target_channels))]
    with torch.no_grad():
        for _ , batch in enumerate(testing_data_loader):
            inp, target = batch[0].to(device), batch[1].to(device)

            prediction = model(inp)
            mse = criterion(prediction, target)

            for x in range(len(target_channels)):
                loss_x = criterion(prediction[:, x, :, :], target[:, x, :, :])
                ssim_x = calc_ssim(prediction[:, x, :, :].view(len(batch[0]), 1, 240, 240), target[:, x, :, :].view(len(batch[0]), 1, 240, 240))
                epoch_loss_indiv[x] += loss_x.item()
                epoch_ssim_indiv[x] += ssim_x

            psnr = 10 * log10(1/mse.item())
            avg_psnr += psnr
            
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
        pred = prediction.cpu()[0]
        target = target.cpu()[0]

        for p, t, c_name in zip(pred, target, target_channels):
            show(epoch, p, t, criterion, images_path, args.model_name, " "+c_name)

        for x in range(len(target_channels)):
            epoch_loss_indiv[x] /= len(testing_data_loader)
            epoch_ssim_indiv[x] /= len(testing_data_loader)
            

        psnr_indiv = list(map(calc_psnr, epoch_loss_indiv))
    return avg_psnr / len(testing_data_loader), psnr_indiv, epoch_ssim_indiv

if args.verbose:
    print("==> Starts to train")
try:
    if cold_start:
        raise Exception
    model, optimizer, start_epoch, stats, start_time = load_checkpoint(model_weights+modelName+".pth", model, optimizer)
    print("Training resumed at epoch ", start_epoch)
except:
    print("training from scratch")
    start_epoch=0

    stats = {
    'model': args.model_name,
    'lr': lr,
    'epoch': [],
    'elapsed_time': [],
    'train_PSNR': [],
    'test_PSNR': [],
    'train_PSNR_t1w': [],
    'test_PSNR_t1w': [],
    'train_ssim_t1w': [],
    'test_ssim_t1w': [],
    'train_PSNR_t1gd': [],
    'test_PSNR_t1gd': [],
    'train_ssim_t1gd': [],
    'test_ssim_t1gd': [],
    'train_PSNR_t2w': [],
    'test_PSNR_t2w': [],
    'train_ssim_t2w': [],
    'test_ssim_t2w': [],
    'train_PSNR_flair': [],
    'test_PSNR_flair': [],
    'train_ssim_flair': [],
    'test_ssim_flair': []
    }
    start_time = time.time()



for epoch in range(start_epoch, nEpochs + 1):

    test_psnr, test_psnr_indiv, test_ssim = test(epoch)
    train_psnr, train_psnr_indiv, train_ssim = train(epoch)

    test_per_modality = {target_channels[x]: test_psnr_indiv[x] for x in range(len(target_channels))}
    train_per_modality = {target_channels[x]: train_psnr_indiv[x] for x in range(len(target_channels))}
    test_ssim_per_modality = {target_channels[x]: test_ssim[x] for x in range(len(target_channels))}
    train_ssim_per_modality = {target_channels[x]: train_ssim[x] for x in range(len(target_channels))}
    print("test_per_mod", test_per_modality)
    print("train_per_mod", train_per_modality)

    stop_time = time.time()
    elapsed_time = np.round(stop_time - start_time, 2)
    stats['elapsed_time'].append(elapsed_time)
    stats['epoch'].append(epoch)
    stats['train_PSNR'].append(train_psnr)
    stats['test_PSNR'].append(test_psnr)
    stats['train_PSNR_t1w'].append(0)
    stats['test_PSNR_t1w'].append(0)
    stats['train_ssim_t1w'].append(0)
    stats['test_ssim_t1w'].append(0)
    stats['train_PSNR_t1gd'].append(0)
    stats['test_PSNR_t1gd'].append(0)
    stats['train_ssim_t1gd'].append(0)
    stats['test_ssim_t1gd'].append(0)
    stats['train_PSNR_t2w'].append(0)
    stats['test_PSNR_t2w'].append(0)
    stats['train_ssim_t2w'].append(0)
    stats['test_ssim_t2w'].append(0)
    stats['train_PSNR_flair'].append(0)
    stats['test_PSNR_flair'].append(0)
    stats['train_ssim_flair'].append(0)
    stats['test_ssim_flair'].append(0)
    for x in target_channels:
        stats['test_PSNR_' + x][-1] = test_per_modality[x]
        stats['train_PSNR_' + x][-1] = train_per_modality[x]
        stats['test_ssim_' + x][-1] = test_ssim_per_modality[x]
        stats['train_ssim_' + x][-1] = train_ssim_per_modality[x]


    ckp = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'stats': stats
    }
    save_checkpoint(ckp, model_weights, args.model_name, epoch)

    df = pd.DataFrame(
        data=stats, 
        columns=['model', 'lr', 'epoch', 'elapsed_time', 'train_PSNR','test_PSNR', 'train_PSNR_t1w', 'test_PSNR_t1w', 'train_ssim_t1w', 'test_ssim_t1w','train_PSNR_t1gd', 'test_PSNR_t1gd', 'train_ssim_t1gd', 'test_ssim_t1gd','train_PSNR_t2w', 'test_PSNR_t2w', 'train_ssim_t2w', 'test_ssim_t2w','train_PSNR_flair', 'test_PSNR_flair''train_ssim_flair', 'test_ssim_flair']
        )
    df.to_csv(stats_path + args.model_name + "_stats.csv")

print("===> Display results")
df = pd.DataFrame(
        data=stats, 
        columns=['model', 'lr', 'epoch', 'elapsed_time', 'train_PSNR','test_PSNR', 'train_PSNR_t1w', 'test_PSNR_t1w', 'train_ssim_t1w', 'test_ssim_t1w','train_PSNR_t1gd', 'test_PSNR_t1gd', 'train_ssim_t1gd', 'test_ssim_t1gd','train_PSNR_t2w', 'test_PSNR_t2w', 'train_ssim_t2w', 'test_ssim_t2w','train_PSNR_flair', 'test_PSNR_flair''train_ssim_flair', 'test_ssim_flair']
        )
df.to_csv(os.path.join("./", args.model_name, "_stats.csv"))

def plot(x, label, linestyle='solid', color='g'):
    t = np.arange(0, len(x), 1)
    plt.plot(t, x, label=label, linestyle=linestyle, color=color)

plt.figure(figsize=(10, 8))
plot(stats['train_PSNR'], "train PSNR", linestyle="dashed", color="g")
plot(stats['test_PSNR'], "test PSNR", linestyle="solid", color="g")
plt.legend()
plt.grid()
plt.xlabel("epochs")
plt.ylabel("PSNR (dB)")
plt.title("Learning Curve" + args.model_name)
plt.show()
plt.savefig(os.path.join('./', args.model_name, "lc.jpg"))