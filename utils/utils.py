import os
import torch
import numpy as np
from PIL import Image
from math import log10
from model.ssim import ssim
import matplotlib.pyplot as plt

def calc_psnr(x):
    return 10 * log10(1/x)

def calc_ssim(pred, target):
    ssim_score  = ssim(pred, target, window_size=11, size_average=True).item()
    return ssim_score

def show(epoch, pred, target, criterion, save_path, model_name, title=""):
    psnr = 10 * log10(1/criterion(pred, target))

    pred = pred.detach().numpy()
    norm = 255./(pred.max()-pred.min())
    pred *= norm
    pred.clip(0, 255)

    target = target.detach().numpy()
    norm = 255./(target.max()-target.min())
    target *= norm
    target.clip(0, 255)


    residual = np.abs(pred - target)
    residual -= residual.min()
    norm = 255./(residual.max()-residual.min())
    residual *= norm
    residual.clip(0, 255)
    
    pred = Image.fromarray(pred, mode='F').convert('L')
    pred.save(os.path.join(save_path, "{}_epoch_{:.4f}_{}dB_{}".format(model_name, epoch, psnr, title + "_prediction.jpg")))

    target = Image.fromarray(np.uint8(target), mode='L')
    target.save(os.path.join(save_path, "{}_epoch_{}_{}".format(model_name, epoch, title + "_target.jpg")))
    residual = Image.fromarray(np.uint8(residual), mode='L')
    residual.save(os.path.join(save_path, "{}_epoch_{}_{}".format(model_name, epoch, title+"_residual.jpg")))

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches((15,10))
    axs[0].imshow(target, cmap='gray')
    axs[0].set_title("Target " + title)
    axs[2].imshow(residual)
    axs[2].set_title("Residual ")
    axs[1].imshow(pred, cmap='gray')
    axs[1].set_title("Prediction " + title + ' psnr= {:.4f} dB'.format(psnr))
    fig.savefig(os.path.join(save_path, "{}_epoch_{}_{}".format(model_name, epoch, title+"_summary.jpg")))
    plt.close()


def save_checkpoint(ckp, ckp_path, model_name, epoch):
    model_out_path = os.path.join(ckp_path, model_name)
    torch.save(ckp, os.path.join(model_out_path, ".pth")) #save ckp
    torch.save(ckp['state_dict'], os.path.join(model_out_path, "model_epoch_{}.pth".format(epoch))) #save model
    print("Checkpoint saved to {}".format(model_out_path))

def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['stats'], checkpoint['stats']['elapsed_time'][-1]