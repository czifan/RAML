from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import torch.nn.functional as F

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes, Cityscapes_Novel
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as Metrics
from torch import Tensor
from typing import Tuple
from sklearn.metrics import f1_score
import cv2

def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss

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
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=10000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: True)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)
    
    parser.add_argument("--ckpt", default="output/final.pth", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
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

    parser.add_argument("--output_dir", type=str, default='output_metric', help="output path")
    
    parser.add_argument("--novel_dir", type=str, default='./novel/', help="novel path")
    
    parser.add_argument("--test_mode", type=str, default='16_3', choices=['16_1','16_3','12','14'],
                        help="test mode")
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

def save_ckpt(batch_idx, model, metric_model, optimizer, scheduler, path):
    """ save current model
    """
    torch.save({
        "batch_idx": batch_idx,
        "model_state": model.module.state_dict(),
        "metric_model": metric_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, path)
    print("Model saved as %s" % path)
def get_spilt_center(feature,target,metric_model,label,device):
    _, H, W, C = feature.shape
    feature = feature.view(H,W,C) # (H*W, M)
    target = target.view(H,W) # (H*W)
    #feature = feature[target==label] # (N, M)
    now_sum = torch.zeros(C,).to(device)
    mask = target == label
    print(mask.shape)
    
    now_center_embedding=[]
    
    mask = mask.cpu().data.numpy()
    mask = mask.astype(np.uint8)

    num_object, connect = cv2.connectedComponents(mask)
    #novel_sum=0
    for k in range(num_object):
        now_connect = (connect == k)[np.newaxis, ...].astype(np.uint8) 
        #now_mask = mask[now_connect]
        now_mask = now_connect * mask
        print(np.sum(now_mask))
        if (np.sum(now_mask)<100): continue
        print(now_mask.shape)
        print(feature.shape)
        now_feature=feature[now_mask==1]
        print(now_feature.shape)
        now_feature=now_feature.view(-1,C)
        now_feature=torch.sum(now_feature,dim=0)/np.sum(now_mask)
        #now_feature=torch.Tensor(now_feature).to(device)
        now_embedding=metric_model.forward_feature(now_feature.unsqueeze(dim=0))[0].detach().cpu().numpy() # (128,)
        now_center_embedding.append(now_embedding)
    return now_center_embedding
def get_all_center(feature,target,metric_model,label):
    _, H, W, C = feature.shape
    feature = feature.view(-1,C) # (H*W, M)
    target = target.flatten() # (H*W)
    feature = feature[target==label] # (N, M)
    feature = torch.sum(feature, dim=0)
    novel_sum = torch.sum(target == label)
    now_center = feature / novel_sum
    now_center_embedding = metric_model.forward_feature(now_center.unsqueeze(dim=0))[0].detach().cpu().numpy() # (128,)
    return now_center_embedding
    
def generate_novel(novel_path_name, unknown_list, model, metric_model, device):
    model.eval()
    metric_model.eval()
    novel_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    center_embedding = {}
    spilt_list=[]
    with torch.no_grad():
        for x in unknown_list: # [13, 14, 15]
            print('generate novel: '+str(x))
            center=[]
            
            novel_dst = Cityscapes_Novel(novel_path=novel_path_name, novel_no=x, transform=novel_transform)
            novel_loader = data.DataLoader(novel_dst, batch_size=1, shuffle=False, num_workers=4)
            novel_sum = 0
            for (image, target) in novel_loader:
                assert image.shape[0] == 1
                image = image.to(device)
                target = target.to(device,dtype=torch.long)
                _,_,feature,_ = model(image)
                target = F.interpolate(target.unsqueeze(dim=1).float(), size=feature.shape[-2:], mode='nearest')[:, 0]
                feature = feature.permute(0, 2, 3, 1) # (1, H, W, M)
                _, H, W, C = feature.shape
                if (x in spilt_list):
                    now_center_embedding=get_spilt_center(feature,target,metric_model,x,device)
                    for now_center in now_center_embedding:
                        center.append(now_center)
                else:
                    now_center_embedding=get_all_center(feature,target,metric_model,label=x)
                    center.append(now_center_embedding)
            #center = center / novel_sum # (M,)
            center=np.array(center)
            print(center.shape)
            
            '''
            random select novel
            
            np.random.seed(333333)
            a = np.random.choice(100,1,False)
            center=center[a]
            print(center.shape)
            '''
            center=np.mean(center,axis=0)
            
            center_embedding[x] = deepcopy(center)
    return center_embedding
'''
def generate_novel(novel_path_name, unknown_list, model, metric_model, device):
    model.eval()
    metric_model.eval()
    novel_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    center_embedding = {}
    with torch.no_grad():
        for x in unknown_list: # [13, 14, 15]
            print('generate novel: '+str(x))
            center=None
            novel_dst = Cityscapes_Novel(novel_path=novel_path_name, novel_no=x, transform=novel_transform)
            novel_loader = data.DataLoader(novel_dst, batch_size=1, shuffle=False, num_workers=4)
            novel_sum = 0
            for (image, target) in novel_loader:
                assert image.shape[0] == 1
                image = image.to(device)
                target = target.to(device,dtype=torch.long)
                _,_,feature,_ = model(image)
                target = F.interpolate(target.unsqueeze(dim=1).float(), size=feature.shape[-2:], mode='nearest')[:, 0]
                feature = feature.permute(0, 2, 3, 1) # (1, H, W, M)
                _, H, W, C = feature.shape
                feature = feature.view(-1, C) # (H*W, M)
                target = target.flatten() # (H*W)
                feature = feature[target==x] # (N, M)
                feature = torch.sum(feature, dim=0)
                if center is None: 
                    center = torch.zeros(C,).to(device)
                center += feature
                novel_sum += torch.sum(target == x)
            center = center / novel_sum # (M,)
            center_embedding[x] = metric_model.forward_feature(center.unsqueeze(dim=0))[0].detach().cpu().numpy() # (128,)
    return center_embedding
'''
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

from copy import deepcopy

def concat_logits(logits, thereshold=100, erode=True, tag=None):
    if (isinstance(tag,list)):
        mask = np.array(tag)
        logits = np.transpose(logits)
        logits = logits * mask
        logits = np.transpose(logits)

    logits = (logits >= 0.5).astype(np.uint8) 
    logits = np.sum(logits,axis=0)
    logits[logits>=1]=1
    mask = logits == 1
    logits = logits.astype(np.uint8)
    if (erode == True):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        logits = cv2.dilate(logits, kernel)
        logits = cv2.erode(logits, kernel)
        
    #print(logits.shape)
    num_object, connect = cv2.connectedComponents(logits)
    region_list = []
    for k in range(1,num_object):
        now_connect = (connect == k)[np.newaxis, ...].astype(np.uint8) 
        #now_sum = np.sum(now_connect)
        #print(now_sum)
        if (np.sum(now_connect) < thereshold):
            mask[connect == k] = 0
            continue
        region_list.append(k) 
    logits = logits * mask
    
    return logits, region_list, connect

def check_novel_logit(opts,model,metric_model,class_no,meta_channel_num,device,beta=0.15):
    model.eval()
    metric_model.eval()
    novel_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    center_embedding = {}
    spilt_list=[]
    channel_tag=[0]*meta_channel_num
    with torch.no_grad():
      
        print('generate novel: '+str(class_no))
        center=[]

        novel_dst = Cityscapes_Novel(novel_path=opts.novel_dir, novel_no=class_no, transform=novel_transform)
        novel_loader = data.DataLoader(novel_dst, batch_size=1, shuffle=False, num_workers=4)
        novel_sum = 0
        for (image, target) in novel_loader:
            assert image.shape[0] == 1
            #image, target = novel_transform(image,target)
            image = image.to(device)
            target = target.to(device,dtype=torch.long)
            output,logit,feature,_ = model(image)
            output = torch.argmax(output[0], dim=0).detach().cpu().numpy()
            mask = target == class_no
            target = F.interpolate(target.unsqueeze(dim=1).float(), size=feature.shape[-2:], mode='nearest')[:, 0]
            #print(target.shape)
            
            #print(mask.shape)
            logit = logit[0, (-meta_channel_num):]
            #print(logit.shape)
            logit = logit * mask
            mask = mask.data.cpu().numpy()
            all_sum=np.sum(mask)
            logit = logit.detach().cpu().numpy()
            logit = (logit >= 0.5).astype(np.uint8) 
            for x in range(logit.shape[0]):
                if (np.sum(logit[x])>all_sum*beta): channel_tag[x]=1
            #print(logit.shape)

            #for x in range(channel_num):
            #print(image.shape)
            #image= denorm(image.detach().cpu().numpy())[0] * 255
            #print(image.shape)
            image = (denorm(image.detach().cpu().numpy())[0] * 255).transpose(1, 2, 0).astype(np.uint8)
            '''
            plt.imshow(image)
            plt.show()
            plt.close()
            _, axarr = plt.subplots(1, logit.shape[0], figsize=(5*logit.shape[0], 5))
         
            for i in range(logit.shape[0]):
                now_logit=cv2.resize(logit[i], output.shape[::-1], interpolation=cv2.INTER_NEAREST)
                axarr[i].imshow(image)
                axarr[i].imshow(now_logit, alpha=0.5)
            plt.show()
            plt.close()
            '''
            '''
            feature = feature.permute(0, 2, 3, 1) # (1, H, W, M)
            
            _, H, W, C = feature.shape
            if (x in spilt_list):
                now_center_embedding=get_spilt_center(feature,target,metric_model,label=x)
                for now_center in now_center_embedding:
                    center.append(now_center)
            else:
                now_center_embedding=get_all_center(feature,target,metric_model,label=x)
                center.append(now_center_embedding)
            '''
        #center = center / novel_sum # (M,)
        '''
        center=np.array(center)
        print(center.shape)
        center=np.mean(center,axis=0)
        center_embedding[x] = deepcopy(center)
        '''
    return channel_tag

def val(opts, model, metric_model, train_loader, val_loader, device,):
    remain_class = 19 - len(Cityscapes.unknown_target)

    metrics16 = StreamSegMetrics(19)
    metrics19 = StreamSegMetrics(19, remain_class)
    model.eval()
    metric_model.eval()
    if opts.save_val_results:
        if not os.path.exists('results_1'):
            os.mkdir('results_1')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
        img_id = 0
    # val_save_dir = os.path.join(opts.output_dir, 'val')
    # os.makedirs(val_save_dir, exist_ok=True)
    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
    #                             std=[0.229, 0.224, 0.225])
    if (opts.test_mode == '16_1'):
        center_embedding = generate_novel(opts.novel_dir, [13], model, metric_model, device) # {13: (128,), 14: (128,), 15: (128,)}    
    else:
        center_embedding = generate_novel(opts.novel_dir, Cityscapes.unknown_target, model, metric_model, device) # {13: (128,), 14: (128,), 15: (128,)}
        
    
    #using when 16+1 setting
    #center_embedding = generate_novel(opts.novel_dir, [13], model, metric_model, device) # {13: (128,), 14: (128,), 15: (128,)}
    
    name=['sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
    meta_channel_num=20-remain_class
    all_tag=[0]*meta_channel_num
    
    if (opts.test_mode == '16_1'):
        for x in [13]:
            novel_tag=check_novel_logit(opts, model,metric_model,x, meta_channel_num=meta_channel_num, device=device)
            for y in range(meta_channel_num):
                if (novel_tag[y]==1): all_tag[y]=1
    else:
        for x in Cityscapes.unknown_target:
            novel_tag=check_novel_logit(opts, model,metric_model,x, meta_channel_num=meta_channel_num, device=device)
            for y in range(meta_channel_num):
                if (novel_tag[y]==1): all_tag[y]=1
    
    #using when 16+1 setting
    '''
    for x in [13]:
        novel_tag=check_novel_logit(opts, model,metric_model,x, meta_channel_num=meta_channel_num, device=device)
        for y in range(meta_channel_num):
            if (novel_tag[y]==1): all_tag[y]=1
    '''
    #all_tag = np.array(all_tag)
    print(all_tag)
    miou_all=[]
    miou_unknown=[]
    for _, (images, labels, labels_true, _, _) in tqdm(enumerate(val_loader)):
        assert images.shape[0] == 1
        with torch.no_grad():
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            labels_true = labels_true.to(device, dtype=torch.long)

            outputs, logits, features, _ = model(images) # outputs: (1, 16, H, W), logits: (1, 20, H, W), features: (1, 256, H/4, W/4)
            known_class = outputs.shape[1]
            h,w=outputs.shape[2],outputs.shape[3]
            #outputs = logits[:,0:known_class,:,:].clone()
            logits = F.interpolate(logits, size=features.shape[-2:], mode='bilinear', align_corners=False) # (1, 20, H/4, W/4)
            features = features[0].detach().cpu().numpy() # (256, H/4, W/4)
            outputs = torch.argmax(outputs, dim=1)[0].detach().cpu().numpy() # (H, W)

            metrics16.update(labels[0].detach().cpu().numpy(), outputs)

            outputs19 = deepcopy(outputs) 
            # in 16 + 3 setting and 16 + 1 setting
            if ('16' in opts.test_mode):
                outputs19[outputs19 == 13] = 16
                outputs19[outputs19 == 14] = 17
                outputs19[outputs19 == 15] = 18
            
            # in 12 + 7 setting 10->12 11,12->10,11
            if ('12' in opts.test_mode):
                outputs19[outputs19 == 11] = 12
                outputs19[outputs19 == 10] = 11
            #in 14 + 5 setting unknown_target = [10,13,14,15,16]
            # 11 -> 10 12 -> 11 17 -> 12 18 -> 13
            if ('14' in opts.test_mode):
                outputs19[outputs19 == 13] = 18
                outputs19[outputs19 == 12] = 17
                outputs19[outputs19 == 11] = 12
                outputs19[outputs19 == 10] = 11
                
            logits = logits[0].detach().cpu().numpy() # (20, H/4, W/4)
            logits = logits[known_class:] # (3, H/4, W/4)
            # concat inference
            
            logits, region, connect = concat_logits(logits, thereshold=250, tag=all_tag)
            for k in region:
                mask = (connect == k)[np.newaxis, ...].astype(np.uint8) # (1, H/4, W/4)     
                embedding = (features * mask).reshape(features.shape[0], -1).sum(axis=-1) # (256,)
                embedding = embedding / np.sum(mask)
                embedding = torch.Tensor(embedding).unsqueeze(dim=0).to(device, dtype=torch.float32) # (1, 256)
                embedding = metric_model.forward_feature(embedding)[0].cpu().detach().numpy() # (128,)
                tmp_key, tmp_cos = None, None 
                for key, value in center_embedding.items():
                    cos = cosine_similarity(embedding, value)
                    if  cos >= 0.8:
                        if tmp_cos is None or cos > tmp_cos:
                            tmp_key = key
                            tmp_cos = cos 
                if tmp_key is not None:
                    mask = cv2.resize(mask[0], outputs19.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    outputs19 = mask * tmp_key + outputs19 * (1 - mask)
                
            '''
            # default inference
            logits = (logits >= 0.5).astype(np.uint8) # (3, H/4, W/4)
            for c in range(logits.shape[0]):
                logit = logits[c] # (H/4, W/4)
                #Hl, Wl = logit.shape
                #logit = cv2.resize(logit, (Wl//4, Hl//4), interpolation=cv2.INTER_NEAREST)
                num_object, connect = cv2.connectedComponents(logit)
                #connect = cv2.resize(connect, (Wl, Hl), interpolation=cv2.INTER_NEAREST)
                for k in range(1, num_object+1):
                    mask = (connect == k)[np.newaxis, ...].astype(np.uint8) # (1, H/4, W/4)
                    if np.sum(mask) < 100: continue 
                    embedding = (features * mask).reshape(features.shape[0], -1).sum(axis=-1) # (256,)
                    embedding = embedding / np.sum(mask)
                    embedding = torch.Tensor(embedding).unsqueeze(dim=0).to(device, dtype=torch.float32) # (1, 256)
                    embedding = metric_model.forward_feature(embedding)[0].cpu().detach().numpy() # (128,)
                    tmp_key, tmp_cos = None, None 
                    for key, value in center_embedding.items():
                        cos = cosine_similarity(embedding, value)
                        if  cos >= 0.75:
                            if tmp_cos is None or cos > tmp_cos:
                                tmp_key = key
                                tmp_cos = cos 
                    if tmp_key is not None:
                        mask = cv2.resize(mask[0], outputs19.shape[::-1], interpolation=cv2.INTER_NEAREST)
                        outputs19 = mask * tmp_key + outputs19 * (1 - mask)
                '''
            #using in 16+3 setting
            if ('16' in opts.test_mode):
                for x in range(13,16):
                    labels_true[labels_true==x]+=103
                    outputs19[outputs19==x]+=103
                    labels_true[labels_true==(x+3)]-=3
                    outputs19[outputs19==(x+3)]-=3
                for x in range(116,119): 
                    labels_true[labels_true==x]-=100
                    outputs19[outputs19==x]-=100
            if (opts.test_mode == '16_1'):
                for x in range(17,19):
                    labels_true[labels_true==x] = 255
            # using in 12 + 7 setting 10->12 11,12->10,11
            if ('12' in opts.test_mode):
                labels_true[labels_true==10] = 112
                outputs19[outputs19==10] =112

                labels_true[labels_true == 11] = 10
                outputs19[outputs19==11] = 10
                labels_true[labels_true == 12] = 11
                outputs19[outputs19 == 12] = 11

                labels_true[labels_true==112] -= 100
                outputs19[outputs19==112] -= 100
            '''
            labels_true[labels_true==10] = 112
            outputs19[outputs19==10] =112
            
            labels_true[labels_true == 11] = 10
            outputs19[outputs19==11] = 10
            labels_true[labels_true == 12] = 11
            outputs19[outputs19 == 12] = 11
            
            labels_true[labels_true==112] -= 100
            outputs19[outputs19==112] -= 100
            '''
            #in 14 + 5 setting unknown_target = [10,13,14,15,16]
            # 11 -> 10 12 -> 11 17 -> 12 18 -> 13
            # 10 -> 14 ,13 ->15
            if ('14' in opts.test_mode):
                labels_true[labels_true == 10] = 114
                outputs19[outputs19 == 10] = 114
                for x in range(13,17):
                    labels_true[labels_true == x] = 100+2+x
                    outputs19[outputs19 == x] = 100+2+x
                for x in range(11,13):
                    labels_true[labels_true == x] = x-1
                    outputs19[outputs19 == x] = x-1
                for x in range(17,19):
                    labels_true[labels_true == x] = x-5
                    outputs19[outputs19 == x] = x-5
                for x in range(114,119):
                    labels_true[labels_true == x] -=100
                    outputs19[outputs19 == x] -=100
            metrics19.update(labels_true[0].detach().cpu().numpy(), outputs19)
            '''
            for x in range(13,16):
                labels_true[labels_true==x]+=103
                outputs19[outputs19==x]+=103
                labels_true[labels_true==(x+3)]-=3
                outputs19[outputs19==(x+3)]-=3
            for x in range(116,119): 
                labels_true[labels_true==x]-=100
                outputs19[outputs19==x]-=100
            '''
            '''
            now_all_IoU = metrics19.get_results()['Mean IoU']
            now_unkown_IoU = metrics19.get_results()['Unknown IoU']
            miou_all.append(now_all_IoU)
            miou_unknown.append(now_unkown_IoU)
            metrics19.reset()
            '''
            #print(labels_true.shape)
            #print(outputs19.shape)
            
            if opts.save_val_results:
                assert images.shape[0] == 1
                target = labels_true[0].detach().cpu().numpy()
                image = images[0].detach().cpu().numpy()
                pred = outputs19
                #pred = pred.reshape(h,w)
                
                image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                target = train_loader.dataset.decode_target(target).astype(np.uint8)
                pred = train_loader.dataset.decode_target(pred).astype(np.uint8)
                #scores = (255 * scores).squeeze().astype(np.uint8)

                Image.fromarray(image).save('results_1/%d_image.png' % img_id)
                Image.fromarray(target).save('results_1/%d_target.png' % img_id)
                Image.fromarray(pred).save('results_1/%d_pred.png' % img_id)
                #Image.fromarray(scores).save('results/%d_scores.png' % img_id)
                
                    # np.save('results/%d_dis_sum.npy' % img_id, dis_sum_map  
                img_id += 1
    score16 = metrics16.get_results()
    score19 = metrics19.get_results()
    now_IoU = score19['Unknown IoU']
    print('16 classes')
    print(metrics16.to_str(score16))
    print()
    print('19 classes')
    print(metrics19.to_str(score19))
    '''
    for x in range(0,100):
        print(x,miou_all[x],miou_unknown[x])
    '''
    return now_IoU

    
def train(opts, model, metric_model, train_loader, val_loader, criterion, optimizer, scheduler, device, printer=print):
    ce_criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    metric_model.train()
    epoch_records = {'f1': []}
    cur_itr = 0
    best_IoU = 0
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    #val_save_dir = os.path.join(opts.output_dir, 'val')
    #os.makedirs(val_save_dir, exist_ok=True)
    while True:
        for batch_idx, (images, labels, labels_true, labels_lst, class_lst) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels_lst = labels_lst.to(device, dtype=torch.long)
            class_lst = class_lst.to(device, dtype=torch.long)
            labels_true = labels_true.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)

            _, _, features, _ = model(images) 
            labels_lst = F.interpolate(labels_lst.float(), size=features.shape[-2:], mode='nearest')
            new_features, new_labels, logits = metric_model(features, labels_lst) 
            cir_loss = criterion(*convert_label_to_similarity(new_features, new_labels)) * 0.1
            ce_loss = ce_criterion(logits, new_labels.long())
            loss = {
                'loss': cir_loss + ce_loss,
                'cir_loss': cir_loss,
                'ce_loss': ce_loss,
            }
            for key, value in loss.items():
                if key not in epoch_records:
                    epoch_records[key] = []
                epoch_records[key].append(value.item())

            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()

            f1 = f1_score(new_labels.detach().cpu().numpy(), 
                    torch.argmax(logits, dim=1).detach().cpu().numpy(), 
                    average='macro')
            epoch_records['f1'].append(f1)

            if batch_idx % 100 == 0:
                context = f"Iters {cur_itr}\t"
                for key, value in epoch_records.items():
                    context += f"{key}: {np.mean(value):.4f}\t"
                printer(context)
                epoch_records = {'f1': []}

            if cur_itr and cur_itr % 1000 == 0:
                now_IoU = val(opts, model, metric_model, train_loader, val_loader, device)
                if (now_IoU > best_IoU):
                    best_IoU = now_IoU
                    save_ckpt(batch_idx, model, metric_model, optimizer, scheduler, os.path.join(opts.output_dir, f'best.pth'))
                print('best IoU :'+str(best_IoU))
                model.eval()
                metric_model.train()

            cur_itr += 1

            if cur_itr >= opts.total_itrs:
                save_ckpt(batch_idx, model, metric_model, optimizer, scheduler, os.path.join(opts.output_dir, f'final.pth'))
                val(opts, model, metric_model, train_loader, val_loader, device)
                return epoch_records

            scheduler.step()

        save_ckpt(batch_idx, model, metric_model, optimizer, scheduler, os.path.join(opts.output_dir, f'{cur_itr}.pth'))

from dropblock import DropBlock2D
class MetricModel(nn.Module):
    def __init__(self, known_class):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128))
        self.classifier = nn.Linear(128, known_class, bias=False)
        self.known_class = known_class
        self.dropblock = DropBlock2D(block_size=3, drop_prob=0.3)

    def forward(self, feature, label_lst):
        # feature: (B, 256, H, W)
        # label_lst: (B, 17, H, W)
        label_lst = label_lst[:, :self.known_class]
        new_feature, new_label = [], []
        for _ in range(self.known_class):
            tmp_label_lst = self.dropblock(label_lst) # (B, 16, H, W)
            for c in range(tmp_label_lst.shape[1]):
                tmp_feature = (feature * tmp_label_lst[:, c:c+1, :, :]).view(feature.shape[0], feature.shape[1], -1) # (B, 256, H*W)
                tmp_feature = tmp_feature.sum(dim=-1) # (B, 256)
                tmp_num = tmp_label_lst[:, c:c+1, :, :].view(tmp_label_lst.shape[0], -1) # (B, H*W)
                tmp_num = tmp_num.sum(dim=-1) # (B,)
                keep_ind = tmp_num != 0
                if keep_ind.shape[0]:
                    tmp_feature = tmp_feature[keep_ind]
                    tmp_num = tmp_num[keep_ind]
                    tmp_feature = tmp_feature / tmp_num.unsqueeze(dim=1) # (B, 256)
                    new_feature.append(tmp_feature)
                    new_label.append(torch.ones(tmp_feature.shape[0])*c)
        new_feature = torch.cat(new_feature, dim=0) # (N, 256)
        new_feature = self.model(new_feature) # (N, 128)
        new_label = torch.cat(new_label, dim=0).to(feature.device) # (N,)
        logit = self.classifier(new_feature) # (N, 16)
        return F.normalize(new_feature), new_label.long(), logit

    def forward_feature(self, feature):
        # feature: (1, 256)
        new_feature = self.model(feature) # (1, 128)
        return F.normalize(new_feature)

def main():
    print(torch.version.cuda)
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19



    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    unknown_num = len(train_dst.unknown_target)
    remain_class = opts.num_classes - unknown_num
    opts.num_classes = remain_class
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

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, metric_dim=opts.metric_dim)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # # Set up metrics
    # metrics = StreamSegMetrics(opts.num_classes)

    #criterion = MyDiceLoss(ignore_index=255).to(device)
    criterion = CircleLoss(m=0.25, gamma=8.0).to(device)
    
    utils.mkdir(opts.output_dir)
    # Restore
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        res = model.load_state_dict(checkpoint["model_state"])
        print(res)
        model = nn.DataParallel(model)
        model.to(device)
        # if opts.continue_training:
        #     optimizer.load_state_dict(checkpoint["optimizer_state"])
        #     scheduler.load_state_dict(checkpoint["scheduler_state"])
        #     print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    for _, param in model.named_parameters():
        param.requires_grad = False

    metric_model = MetricModel(remain_class).to(device)
    optimizer = torch.optim.SGD(metric_model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    if (opts.test_only):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        metric_model.load_state_dict(checkpoint["metric_model"])
        val(opts, model, metric_model, train_loader, val_loader, device)
        return 
        #res = model.load_state_dict(checkpoint["model_state"])
        print(res)
        #model = nn.DataParallel(model)
        #model.to(device)
        
    train(opts, model, metric_model, train_loader, val_loader, criterion, optimizer, scheduler, device, printer=print)

        
if __name__ == '__main__':
    main()
