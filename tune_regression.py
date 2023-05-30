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
import scipy
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
import torch

import warnings
from model_regression import CountRegression, vgg11_model, regression_model
import util.misc as misc
import cv2
import pickle

def get_args_parser():
    parser = argparse.ArgumentParser('regression-tunning', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='regression_model', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=8e-4,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=8e-5, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--data_path', default='/root/CAC_DIP2023-exemplar-resnet/data_tuple.pkl', type=str,
                        help='dataset path')
    parser.add_argument('--load_path', default='/root/CAC_DIP2023-exemplar-resnet/model_regression_aug.pth', type=str,)
    parser.add_argument('--output_dir', default='./data/out/fim6_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from checkpoint')

    # Training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=6, type=int)
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

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'


class TuneData(Dataset):
    def __init__(self,datapath):
        self.file=open(datapath,'rb')
        self.data=pickle.load(self.file)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data=torch.from_numpy(self.data[idx][0])
        label=torch.tensor(self.data[idx][1])
        return data,label

wandb_run = None
log_writer = None


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = TuneData(args.data_path)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
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


    # define the model
    model = CountRegression()
    model.load_state_dict(torch.load(args.load_path))
    model.to(device)
    criterion = nn.SmoothL1Loss() 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.3*args.epochs,0.6*args.epochs,0.9*args.epochs], gamma=0.1)
    best_loss = float('inf')
    best_model_state= None

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for inputs,labels in data_loader_train:
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            outputs = model(inputs)
            labels=labels.unsqueeze(1)
            #print(outputs.shape, labels.shape)
            #labels=torch.sum(inputs,dim=(1,2))
            #labels=labels.unsqueeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad() 

            loss.backward()  
            #print(loss)
            optimizer.step()

            
            total_loss += loss.item()
        scheduler.step()
        if best_loss > total_loss/len(data_loader_train):
            best_loss = total_loss/len(data_loader_train)
            best_model_state = model.state_dict()
        print(f'Epoch {epoch} - Loss: {total_loss/len(data_loader_train)}') 
        torch.cuda.synchronize()

        loss_value_reduce = misc.all_reduce_mean(loss.item())

        if misc.is_main_process():
            if wandb_run is not None:
                log = {"Train/Loss":total_loss/len(data_loader_train),'epoch':epoch,'lr':optimizer.param_groups[0]['lr']}
                wandb.log(log,
                            commit=True )

        # if args.output_dir and misc.is_main_process():
        #     # if log_writer is not None:
        #     #     log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")
 

    torch.save(best_model_state, 'model_regression_tune.pth') 

    wandb.run.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # load data from pkl file
    data_path = Path(args.data_path)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


