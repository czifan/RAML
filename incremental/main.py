from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import torch.nn.functional as F

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sklearn.metrics as Metrics
from torch import Tensor
from typing import Tuple
def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='../data/cityscapes',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=256,
                        help="num classes (default: None)")
    parser.add_argument("--metric_dim", type=int, default=None,
                        help="num classes (default: None)")
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_metirc_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3plus_metirc_resnet101'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--finetune", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"") 
    parser.add_argument("--total_itrs", type=int, default=30000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=1000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: True)')
    parser.add_argument("--batch_size", type=int, default=6,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0,1',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--name", type=str, default='',help="download datasets")

    parser.add_argument("--output_dir", type=str, default='output', help="output path")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
            
    return parser

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class MyDiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.dice_criterion = BinaryDiceLoss()
        self.ignore_index = ignore_index

    def forward(self, logit, label_lst, class_lst):
        loss = 0.0
        for b in range(logit.shape[0]):
            logit_b = logit[b][torch.where(class_lst[b] != self.ignore_index)]
            label_lst_b = label_lst[b][torch.where(class_lst[b] != self.ignore_index)] 
            if logit_b.shape[0]:
                loss += self.dice_criterion(logit_b, label_lst_b)
        return loss / logit.shape[0]

class CDiceLoss(nn.Module):
    def __init__(self, known_class=16, ignore_index=255):
        super().__init__()
        self.dice_criterion = BinaryDiceLoss()
        self.bce_criterion = nn.BCELoss()
        self.ignore_index = ignore_index
        self.class_num=known_class
        print('finetune with '+str(known_class)+" classes")

    def forward(self, logit, label_lst, class_lst):
        loss1 = torch.FloatTensor([0.0]).to(logit.device)
        for i in range(self.class_num):
            loss1 += (self.dice_criterion(logit[:, i], label_lst[:, i]) + self.bce_criterion(logit[:, i], label_lst[:, i].float()))
        loss1 /= self.class_num

        loss2 = 0.0
        for i in range(self.class_num, logit.shape[1]):
            loss2 += -torch.log((torch.mean(logit[:, i]) * 50).clamp(0, 1))
        loss2 /= (logit.shape[1] - self.class_num)

        loss3 = 0.0
        num3 = 0
        for i in range(logit.shape[1]):
            for j in range(logit.shape[1]):
                if i == j: continue 
                dice_loss = self.dice_criterion(logit[:, i], logit[:, j])
                loss3 += (1.0 - dice_loss)
                num3 += 1
        loss3 = loss3 / num3

        loss = (loss1 + loss2 + loss3) * 0.1
        return {
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
        }

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst

def save_ckpt(batch_idx, model, optimizer, scheduler, path):
    """ save current model
    """
    torch.save({
        "batch_idx": batch_idx,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, path)
    print("Model saved as %s" % path)

def visualize(image, label, logit, label_lst, class_lst, save_path=None, denorm=None):
    # logit: (256, H, W)
    if not isinstance(image, np.ndarray):
        image = image.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        logit = logit.detach().cpu().numpy()
        label_lst = label_lst.detach().cpu().numpy()
        class_lst = class_lst.detach().cpu().numpy()
    if denorm:
        image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)

    _, axarr = plt.subplots(2, (1+logit.shape[0]), figsize=(5*(1+logit.shape[0]), 10))
    axarr[0][0].imshow(image)
    label[label == 255] = 0
    axarr[1][0].imshow(label)

    for i in range(logit.shape[0]):
        if i < label_lst.shape[0]:
            axarr[0][1+i].imshow(label_lst[i])
        axarr[1][i+1].imshow((logit[i] >= 0.5).astype(np.uint8))

    # _, axarr = plt.subplots(16, 32, figsize=(40, 20))
    # for i in range(label.shape[0]):
    #     axarr[i//16][(i%16)*2].imshow(label[i])
    #     axarr[i//16][(i%16)*2].set_xticks([])
    #     axarr[i//16][(i%16)*2].set_yticks([])
    # for i in range(logit.shape[0]):
    #     axarr[i//16][(i%16)*2+1].imshow((logit[i] >= 0.5).astype(np.uint8))
    #     axarr[i//16][(i%16)*2+1].set_xticks([])
    #     axarr[i//16][(i%16)*2+1].set_yticks([])

#     label[label == 255] = 19
#     C = logit.shape[0]
#     logit = np.argmax(logit, axis=0)
#     mask = np.zeros_like(logit)
#     for c in range(C):
#         t = class_lst[c]
#         if t == 255: t = 19 
#         temp = (logit == c).astype(np.uint8)
#         mask = np.ones_like(logit) * t * temp + mask * (1 - temp)
#     _, axarr = plt.subplots(1, 3, figsize=(15, 5))
#     axarr[0].imshow(image)
#     axarr[1].imshow(label)
#     axarr[2].imshow(mask)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def val(opts, model, val_loader, device):
    metrics = StreamSegMetrics(19)
    val_save_dir = os.path.join(opts.output_dir, 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    model.eval()
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    for batch_idx, (images, labels, _, _, _) in tqdm(enumerate(val_loader)):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        outputs, _, _, _ = model(images)
        outputs = torch.argmax(outputs, dim=1)[0].detach().cpu().numpy() # (H, W)
        #print(labels.shape, outputs.shape)
        metrics.update(labels[0].detach().cpu().numpy(), outputs)
    
    score = metrics.get_results()
    print(str(opts.num_classes)+' classes')
    print(metrics.to_str(score))
def train_stage1(opts, model, train_loader, val_loader, criterion, optimizer, scheduler, device, printer=print):
    ce_criterion = utils.CrossEntropyLoss(ignore_index=255, size_average=True)
    #l2_criterion = nn.MSELoss().to(device)
    model.train()
    epoch_records = {}
    cur_itr = 0
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    val_save_dir = os.path.join(opts.output_dir, 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    while True:
        for batch_idx, (images, labels, labels_true, labels_lst, class_lst) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels_lst = labels_lst.to(device, dtype=torch.long)
            class_lst = class_lst.to(device, dtype=torch.long)
            labels_true = labels_true.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)

            outputs, _, _, res_images = model(images) 
            #logits = torch.sigmoid(logits)
            # loss = criterion(logits, labels_lst[:, :masks.shape[1]] * masks, class_lst)
            #loss = criterion(logits, labels_lst, class_lst)
            loss_seg = ce_criterion(outputs, labels, None)
            #masks = ((labels.unsqueeze(dim=1)) != 255).float()
            #loss_l2 = l2_criterion(res_images, images) * 0.01
            #loss['loss'] += (loss_seg + loss_l2)
            ##loss['loss_l2'] = loss_l2
            if ("seg" not in epoch_records): epoch_records["seg"]=[]
            epoch_records["seg"].append(loss_seg.cpu().data.numpy())
            #loss_ce = ce_criterion(outputs, labels, None)
            #epoch_records['loss_ce'].append(loss_ce.item())
            #loss = loss + loss_ce

            optimizer.zero_grad()
            loss_seg.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                context = f"Iters {cur_itr}\t"
                for key, value in epoch_records.items():
                    context += f"{key}: {np.mean(value):.4f}\t"
                printer(context)
                epoch_records = {}

            if cur_itr % 1000 == 0:
                val(opts, model, val_loader, device)
                #for _, (images, labels, labels_true, labels_lst, class_lst) in enumerate(val_loader):
                #    if np.random.uniform(0, 1) < 0.9: continue 
                '''
                for b in range(images.shape[0]):
                    visualize(images[b], labels_true[b], logits[b], labels_lst[b], class_lst[b], save_path=os.path.join(val_save_dir, f'{cur_itr}_{b}.png'), denorm=denorm)
                #    break 
                '''
                model.train()

            cur_itr += 1

            if cur_itr >= opts.total_itrs:
                save_ckpt(batch_idx, model, optimizer, scheduler, os.path.join(opts.output_dir, f'final.pth'))
                return epoch_records

            scheduler.step()

        save_ckpt(batch_idx, model, optimizer, scheduler, os.path.join(opts.output_dir, f'{cur_itr}.pth'))
def train(opts, model, train_loader, val_loader, criterion, optimizer, scheduler, device, printer=print):
    ce_criterion = utils.CrossEntropyLoss(ignore_index=255, size_average=True)
    l2_criterion = nn.MSELoss().to(device)
    model.train()
    epoch_records = {}
    cur_itr = 0
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    val_save_dir = os.path.join(opts.output_dir, 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    while True:
        for batch_idx, (images, labels, labels_true, labels_lst, class_lst) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels_lst = labels_lst.to(device, dtype=torch.long)
            class_lst = class_lst.to(device, dtype=torch.long)
            labels_true = labels_true.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)

            outputs, logits, _, res_images = model(images) 
            #logits = torch.sigmoid(logits)
            # loss = criterion(logits, labels_lst[:, :masks.shape[1]] * masks, class_lst)
            loss = criterion(logits, labels_lst, class_lst)
            loss_seg = ce_criterion(outputs, labels, None)
            masks = ((labels.unsqueeze(dim=1)) != 255).float()
            loss_l2 = l2_criterion(res_images, images) * 0.01
            loss['loss'] += loss_l2
            loss['loss'] += loss_seg
            loss['loss_seg'] = loss_seg
            loss['loss_l2'] = loss_l2
            for key, value in loss.items():
                if key not in epoch_records:
                    epoch_records[key] = []
                epoch_records[key].append(value.item())
            #loss_ce = ce_criterion(outputs, labels, None)
            #epoch_records['loss_ce'].append(loss_ce.item())
            #loss = loss + loss_ce

            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                context = f"Iters {cur_itr}\t"
                for key, value in epoch_records.items():
                    context += f"{key}: {np.mean(value):.4f}\t"
                printer(context)
                epoch_records = {}

            if cur_itr % 500 == 0:
                val(opts, model, val_loader, device)
                #for _, (images, labels, labels_true, labels_lst, class_lst) in enumerate(val_loader):
                #    if np.random.uniform(0, 1) < 0.9: continue 
                for b in range(images.shape[0]):
                    visualize(images[b], labels_true[b], logits[b], labels_lst[b], class_lst[b], save_path=os.path.join(val_save_dir, f'{cur_itr}_{b}.png'), denorm=denorm)
                #    break 
                model.train()

            cur_itr += 1

            if cur_itr >= opts.total_itrs:
                save_ckpt(batch_idx, model, optimizer, scheduler, os.path.join(opts.output_dir, f'final.pth'))
                return epoch_records

            scheduler.step()

        save_ckpt(batch_idx, model, optimizer, scheduler, os.path.join(opts.output_dir, f'{cur_itr}.pth'))
        # if batch_idx % 10 == 0:
        #     val(opts, model, val_loader, device)
        #     model.train()

import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except Exception:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)
    
def main():
    print(torch.version.cuda)
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    opts.num_classes = 256

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=8)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=8)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3plus_metirc_resnet101': network.deeplabv3plus_metirc_resnet101
    }
    remain_class = 19 - len(train_dst.unknown_target)
    print('class num : '+str(remain_class))
    opts.num_classes=remain_class
    model = model_map[opts.model](num_classes=remain_class, output_stride=opts.output_stride, metric_dim=opts.metric_dim, finetune=False)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # # Set up metrics
    # metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    if (opts.finetune):
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    #criterion = MyDiceLoss(ignore_index=255).to(device)
    
    criterion = CDiceLoss(remain_class).to(device)
    
    utils.mkdir(opts.output_dir)
    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint["model_state"]
        for key in checkpoint_state_dict:
            if model_state_dict[key].shape != checkpoint_state_dict[key].shape:
                print(key)
                continue
            model_state_dict[key] = checkpoint_state_dict[key]
        model.load_state_dict(model_state_dict)
        #model.load_state_dict(checkpoint["model_state"])
        #model = nn.DataParallel(model)
        device_ids=list(map(int, opts.gpu_id.split(',')))
        #torch.cuda.set_device(device_ids[0])
        print(device_ids)
        #model = nn.DataParallel(model, device_ids=list(map(int, opts.gpu_id.split(','))))
        model = BalancedDataParallel(2, model, dim=0, device_ids=[0,1])
        #model = BalancedDataParallel(2, model, dim=0, device_ids=list(map(int, opts.gpu_id.split(','))))
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        #model = nn.DataParallel(model)
        model = BalancedDataParallel(2, model, dim=0, device_ids=[0,1])
        model.to(device)
    if (opts.finetune):
        train(opts, model, train_loader, val_loader, criterion, optimizer, scheduler, device, printer=print)
    else:
        train_stage1(opts, model, train_loader, val_loader, None, optimizer, scheduler, device, printer=print)
        
if __name__ == '__main__':
    main()
