import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import sys
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchvision
import wandb

import torch.optim as optim

from torchvision import transforms
import scipy.ndimage as ndimage
import torch.nn as nn

import warnings
from model_regression import CountRegression
import util.misc as misc
import cv2

def get_args_parser():
    parser = argparse.ArgumentParser('regression-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='regression_model', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--data_path', default='./datasets/FSC/FSC147_384_V2', type=str,
                        help='dataset path')
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str,
                        help='data split json file')
    parser.add_argument('--density_dir', default='gt_density_map_adaptive_384_VarV2', type=str,
                        help='ground truth directory')
    parser.add_argument('--output_dir', default='./data/out/fim6_dir',
                        help='path where to save, empty for no saving')
    # parser.add_argument('--device', default='cuda',
    #                     help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from checkpoint')

    # Training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Logging parameters
    parser.add_argument('--log_dir', default=None, type=str,
                        help='path where to tensorboard log')
    parser.add_argument("--title", default="CounTR-CLIP", type=str)
    parser.add_argument("--wandb", default=None, type=str)
    parser.add_argument("--team", default="fdudip", type=str)
    parser.add_argument("--wandb_id", default=None, type=str)

    return parser

#os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

class ResizeSomeImage(object):
    def __init__(self, data_path=Path('./datasets/FSC/FSC147_384_V2/')):
        self.im_dir = data_path / 'images_384_VarV2'
        data_split_file = data_path / 'Train_Test_Val_FSC_147.json'


        with open(data_split_file) as f:
            data_split = json.load(f)

        self.train_set = data_split['train']

class ResizeDensity(ResizeSomeImage):
    """
    Resize the DENSITY so that:
        1. Density is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is possibly preserved
    """

    def __init__(self, data_path=Path('./datasets/FSC147/'), MAX_HW=384):
        super().__init__(data_path)
        self.max_hw = MAX_HW

    def __call__(self, sample):
        density =  sample['gt_density']


        new_H = 384
        new_W = 384
        resized_density = cv2.resize(density, (new_W, new_H))

        # density shape[384,384]
        sample = { 'gt_density': resized_density}

        return sample

def transform_train(data_path: Path, MAX_HW: int):
    return transforms.Compose([ResizeDensity(data_path, MAX_HW)])

class TrainData(Dataset):
    def __init__(self):
        self.density = data_split['train']
        random.shuffle(self.density)
        # print(density_dir)
        self.density_dir = density_dir
        self.TransformTrain = transform_train(data_path,384)

    def __len__(self):
        return len(self.density)

    def __getitem__(self, idx):
        density_id = self.density[idx]

        density_path = self.density_dir / (density_id.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')

        sample = {'gt_density': density}
        sample = self.TransformTrain(sample)
        return sample['gt_density']
    

class ValData(Dataset):
    def __init__(self, ):
        self.density = data_split['val']
        self.density_dir = density_dir
        self.TransformTrain = transform_train(data_path,384)
    def __len__(self):
        return len(self.density)

    def __getitem__(self, idx):
        density_id = self.density[idx]

        density_path = self.density_dir / (density_id.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')

        sample = {'gt_density': density}
        sample = self.TransformTrain(sample)
        return sample['gt_density']

class TestData(Dataset):
    def __init__(self, ):
        self.density = data_split['test']
        self.density_dir = density_dir
        self.TransformTrain = transform_train(data_path,384)
    def __len__(self):
        return len(self.density)

    def __getitem__(self, idx):
        density_id = self.density[idx]

        density_path = self.density_dir / (density_id.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')

        sample = {'gt_density': density}
        sample = self.TransformTrain(sample)
        return sample['gt_density']

wandb_run = None
log_writer = None

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = TrainData()
    dataset_test = ValData()

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0:
        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None
        if args.wandb is not None:
            wandb_run = wandb.init(
                config=args,
                resume="allow",
                project=args.wandb,
                name=args.title,
                entity=args.team,
                tags=["CounTR", "finetuning"],
                id=args.wandb_id,
            )
        else:
            wandb_run = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    model = CountRegression()

    # model.to(device)
    criterion = nn.SmoothL1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    best_model_state= None

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for inputs in data_loader_train:
            # print(inputs.shape)
            outputs = model(inputs.unsqueeze(1))
            labels=torch.sum(inputs,dim=(1,2))
            labels=labels.unsqueeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad() 

            loss.backward()  

            optimizer.step()
            
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch {epoch} - Loss: {total_loss/len(train_loader)}') 
        #torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())

        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss.item())
        if misc.is_main_process():
            if wandb_run is not None:
                log = {"Train/Loss":loss.item()}
                wandb.log(log, step=epoch_1000x,
                            commit=True if data_iter_step == 0 else False)

        model.eval()
        val_loss = 0; val_acc = 0 
        for inputs in data_loader_test:
            outputs = model(inputs).item()
            labels=torch.sum(inputs).item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  
            
        if (epoch) % 10 == 0:
            print(f'Epoch {epoch} - Val Loss: {val_loss}') 
        if misc.is_main_process():
            if wandb_run is not None:
                log = {"Val/MAE with text": val_mae / len(data_loader_test),
                        "Val/RMSE with text": (val_rmse / len(data_loader_test)) ** 0.5,
                        "Val/MAE without text": val_zs_mae / len(data_loader_test),
                        "Val/RMSE without text": (val_zs_rmse / len(data_loader_test)) ** 0.5}
                wandb.log(log, step=epoch_1000x)
        if args.output_dir and misc.is_main_process():
            # if log_writer is not None:
            #     log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    model.load_state_dict(best_model_state)  

    torch.save(model.state_dict(), 'model_regression.pth') 

    wandb.run.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # load data from FSC147
    data_path = Path(args.data_path)
    data_split_file = data_path / args.data_split_file
    density_dir = data_path / args.density_dir

    with open(data_split_file) as f:
        data_split = json.load(f)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


