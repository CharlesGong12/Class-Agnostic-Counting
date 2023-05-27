import json
from pathlib import Path

import numpy as np
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from PIL import Image
import open_clip

import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

MAX_HW = 224
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

tokenizer = open_clip.get_tokenizer('ViT-B-32')
clip_enc, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

class ResizeSomeImage(object):
    def __init__(self, data_path=Path('./data/FSC147/')):
        self.im_dir = data_path / 'images_384_VarV2'
        anno_file = data_path / 'annotation_FSC147_384.json'
        data_split_file = data_path / 'Train_Test_Val_FSC_147.json'
        class_file = data_path / 'ImageClasses_FSC147.txt'

        with open(anno_file) as f:
            self.annotations = json.load(f)

        with open(data_split_file) as f:
            data_split = json.load(f)

        self.class_dict = {}
        with open(class_file) as f:
            for line in f:
                key = line.split()[0]
                val = ' '.join(line.split()[1:])
                self.class_dict[key] = val
        self.train_set = data_split['train']


class ResizePreTrainImage(ResizeSomeImage):
    """
    Resize the image so that:
        1. Image is equal to 224*224
        2. The new height and new width are divisible by 16
        3. The aspect ratio is preserved
    Density and boxes correctness not preserved(crop and horizontal flip)
    """

    def __init__(self, data_path=Path('./data/FSC147/'), MAX_HW=384):
        super().__init__(data_path)
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, density = sample['image'], sample['gt_density']

        W, H = image.size

        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        '''scale_factor = float(256)/ H
        new_H = 16*int(H*scale_factor/16)
        new_W = 16*int(W*scale_factor/16)'''
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))
        orig_count = np.sum(density)
        new_count = np.sum(resized_density)

        if new_count > 0:
            resized_density = resized_density * (orig_count / new_count)

        resized_image = PreTrainNormalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image': resized_image, 'boxes': boxes, 'gt_density': resized_density}
        return sample


class ResizeTrainImage(ResizeSomeImage):
    """
    Resize the image so that:
        1. Image is equal to 224 * 224
        2. The new height and new width are divisible by 16
        3. The aspect ratio is possibly preserved
    Density map is cropped to have the same size(and position) with the cropped image
    Exemplar boxes may be outside the cropped area.
    Augmentation including Gaussian noise, Color jitter, Gaussian blur, Random affine, Random horizontal flip and Mosaic (or Random Crop if no Mosaic) is used.
    """

    def __init__(self, data_path=Path('./data/FSC147/'), MAX_HW=224):
        super().__init__(data_path)
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, density, dots, im_id, m_flag = sample['image'], sample['gt_density'], \
            sample['dots'], sample['id'], sample['m_flag']

        W, H = image.size

        new_H=new_W=224
        # new_H = 16 * int(H / 16)
        # new_W = 16 * int(W / 16)
        scale_factor = float(new_W) / W
        resized_image = transforms.Resize((new_H, new_W))(image)
        # print(type(resized_image))
        resized_image=transforms.ToTensor()(resized_image)
        # print(np.sum((density*60)>=1))
        resized_density = cv2.resize(density, (new_H,new_W))    # both of image and density are resized to 224*224

        # Augmentation probability
        aug_p=random.random()
        aug_flag=0
        if aug_p<0.1:   # Gaussian noise
            aug_flag=1
        if aug_p>=0.1 and aug_p<0.2:    #flip
            aug_flag=2
        if aug_p>=0.2 and aug_p<0.3:    #both
            aug_flag=3  # both
        
        # Gaussian noise
        if aug_flag==1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            resized_image = resized_image + noise
            resized_image = torch.clamp(resized_image, 0, 1)
            resized_image=Augmentation(resized_image)
        
        # flip
        if aug_flag==2:
            resized_image=TF.hflip(resized_image)
            resized_density=cv2.flip(resized_density,1)
        
        # both
        if aug_flag==3:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            resized_image = resized_image + noise
            resized_image = torch.clamp(resized_image, 0, 1)
            resized_image=Augmentation(resized_image)
            resized_image=TF.hflip(resized_image)
            resized_density=cv2.flip(resized_density,1)


        # re_image = resized_image
        # resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]), dtype='float32')
        # for i in range(dots.shape[0]):
        #     resized_density[min(new_H - 1, int(dots[i][1]))][min(new_W - 1, int(dots[i][0] * scale_factor))] = 1

        
        # Density map scale up
        reresized_density = resized_density * 60
        reresized_density=torch.from_numpy(reresized_density)
        # print((reresized_density>=1).sum())
        # Word vector
        wv = tokenizer([self.class_dict[im_id]])
        #print( reresized_density)
        # boxes shape [3,3,64,64], image shape [3,224,224], density shape[224,224]
        sample = {'image': resized_image, 'word_vector': wv, 'class_name': self.class_dict[im_id], 'gt_density': reresized_density, 'm_flag': m_flag}

        return sample


PreTrainNormalize = transforms.Compose([
    transforms.RandomResizedCrop(384, scale=(0.2, 1.0), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
])

TTensor = transforms.Compose([
    transforms.ToTensor(),
])

Augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.GaussianBlur(kernel_size=(7, 9))
])

Normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
])

def transform_train(data_path: Path):
    return transforms.Compose([ResizeTrainImage(data_path, MAX_HW)])


def transform_pre_train(data_path: Path):
    return transforms.Compose([ResizePreTrainImage(data_path, MAX_HW)])
