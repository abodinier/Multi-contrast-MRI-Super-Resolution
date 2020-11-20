import torch
import random
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
from os import makedirs, remove
import torch.utils.data as data
from os.path import exists, join, basename
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

def is_image_file(filename):
    return filename.endswith(".npy")

def convert2pilImage(x):
    return Image.fromarray(x, mode='F').convert('L')

def load_img(filepath):
    data = np.load(filepath, allow_pickle=True)
    img = [ data[0][i, :, :] for i in range(4) ]
    data = {'flair' : img[0], 't1w' : img[1], 't1gd' : img[2], 't2w' : img[3], 'mask' : data[1]}
    return data

def make_patch(patch_size, inp, target, mask):
    h,w = patch_size
    # samples a patch from an image, centered on tumor if possible
    is_valid = False
    while not is_valid:
        try:
            p = random.choice(np.argwhere(mask == 1)) #center on random tumor pixel
        except:
            p = random.choice(np.argwhere(mask == 0)) #if there's no tumor on the image inp
            no_tumor = True

        box = [p[0]-(h//2), p[0]+((h+1)//2), p[1]-(w//2), p[1]+((w+1)//2)]

        if box[0] > 0 and box[1] < 240 and box[2] > 0 and box[3] < 240:
            inp_patch = [ i[ box[0]:box[1], box[2]:box[3] ] for i in inp ]
            targ_patch = [ i[ box[0]:box[1], box[2]:box[3] ] for i in target ]
            mask_patch = mask[ box[0]:box[1], box[2]:box[3] ]
            #if (mask_patch == 1).sum() > 5 or no_tumor:
            #    is_valid = True
            is_valid = True
            inp_patch = [ convert2pilImage(x) for x in inp_patch ]
            targ_patch = [ convert2pilImage(x) for x in targ_patch ]

    return inp_patch, targ_patch, mask_patch

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, modalities, channelTarget, input_transform=None, target_transform=None, patch_size=[64,64]):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.modalities = modalities
        self.channelTarget = channelTarget
        self.patch_size = patch_size

    def __getitem__(self, index):
        # extracts all the modalities :
        data = load_img(self.image_filenames[index])

        inp = [ data[x] for x in self.modalities ]
        target = [ data[x].copy() for x in self.channelTarget ]
        mask = data['mask']

        # extracts patches among the image if size ≠ (240, 240)
        h, w = self.patch_size
        if h != 240 and w != 240:
            in_patch, targ_patch, mask_patch = make_patch(self.patch_size, inp, target, mask)
        else :
            in_patch = [ convert2pilImage(x) for x in inp ]
            targ_patch = [ convert2pilImage(x) for x in target]
            mask_patch = mask

        # Pre processing pipeline
        mask_patch = ToTensor()(mask_patch).float()
        if self.input_transform:
            in_patch = [self.input_transform(x).float() for x in in_patch]
        if self.target_transform:
            targ_patch = [self.target_transform(x).float() for x in targ_patch]

        # Stacks it all in tensors:
        h_in, w_in = in_patch[0].size()[1], in_patch[0].size()[2]
        h_tar, w_tar = targ_patch[0].size()[1], targ_patch[0].size()[2]

        in_patch = torch.stack(in_patch, dim=1).view(len(self.modalities), h_in, w_in)
        targ_patch = torch.stack(targ_patch, dim=1).view(len(self.channelTarget), h_tar, w_tar)
        
        return in_patch, targ_patch, mask_patch

    def __len__(self):
        return len(self.image_filenames)


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor, early_upsampling=False):
    processing = []
    processing.append(Resize(crop_size // upscale_factor))
    if early_upsampling:
        processing.append(Resize(crop_size))
    processing.append(ToTensor())

    return Compose(processing)


def target_transform(crop_size):
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor, modalities, channelTarget, training_path, patch_size = [64, 64], early_upsampling=False):
    crop_size = calculate_valid_crop_size(240, upscale_factor)
    return DatasetFromFolder(training_path,
                            modalities, 
                            channelTarget,
                            input_transform=input_transform(crop_size, upscale_factor, early_upsampling),
                            target_transform=target_transform(crop_size), patch_size=patch_size)


def get_test_set(upscale_factor, modalities, channelTarget, test_path, patch_size = [64, 64], early_upsampling=False):
    crop_size = calculate_valid_crop_size(240, upscale_factor)
    return DatasetFromFolder(test_path,
                            modalities, 
                            channelTarget,
                            input_transform=input_transform(crop_size, upscale_factor, early_upsampling),
                            target_transform=target_transform(crop_size), patch_size=patch_size)
