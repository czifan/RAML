from datasets.cityscapes_novel import Cityscapes_Novel
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import torch.nn.functional as F

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, Cityscapes_Novel
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from collections import namedtuple
from utils import colorEncode

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as Metrics
from sklearn.mixture import GaussianMixture
from statsmodels.distributions.empirical_distribution import ECDF
import joblib
import json
from sklearn import manifold
import queue

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
classes = [
    CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
]

train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([255, 255, 255])
colors = np.array(train_id_to_color)
colors = np.uint8(colors)

from dropblock import DropBlock2D
class MetricModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128))
        self.classifier = nn.Linear(128, 10, bias=False)
        self.dropblock = DropBlock2D(block_size=3, drop_prob=0.3)

    def forward(self, feature, label_lst):
        # feature: (B, 256, H, W)
        # label_lst: (B, 17, H, W)
        label_lst = label_lst[:, :10]
        new_feature, new_label = [], []
        for _ in range(10):
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
        
def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_metirc_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3plus_embedding_resnet101','deeplabv3plus_metirc_resnet101'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--metric_dim", type=int, default=None,
                        help="num classes (default: None)")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--center", action='store_true', default=False,
                        help="use center checkpoint")
    parser.add_argument("--center_checkpoint", type=str, default='./center.npy',
                        help="use center checkpoint")



    parser.add_argument("--ckpt", default=None, type=str,
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

def Normalization(x):
    min_value = np.min(x)
    max_value = np.max(x)
    return (x - min_value) / (max_value - min_value)

def Certainty(x, ecdf, thre1, thre2, mean, cov):
    x = ecdf(x)
    # res = x
    # res[res>0.2] = 1
    threshold = ecdf(thre1)
    coefficient = 50
    res = 1 / (1 + np.exp(-coefficient * (x - threshold)))

    return res

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
        novel_dst =  Cityscapes(root=opts.data_root,
                             split='train', transform=val_transform)
    return train_dst, val_dst, novel_dst

def Coefficient_map(x, thre):
    lamda = 20
    return 1 / (1 + np.exp(lamda * (x - thre)))

def val(opts, model, metric_model, train_loader, val_loader, device):
    metrics16 = StreamSegMetrics(19)
    metrics19 = StreamSegMetrics(19)
    model.eval()
    metric_model.eval()
    # val_save_dir = os.path.join(opts.output_dir, 'val')
    # os.makedirs(val_save_dir, exist_ok=True)
    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
    #                             std=[0.229, 0.224, 0.225])
    center_embedding = generate_novel('novel', Cityscapes.unknown_target, model, metric_model, device) # {13: (128,), 14: (128,), 15: (128,)}
    #center_embedding = align_embedding(opts, model, metric_model, train_loader, device, center_embedding)
    for _, (images, labels, labels_true, _, _) in tqdm(enumerate(val_loader)):
        assert images.shape[0] == 1
        with torch.no_grad():
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            labels_true = labels_true.to(device, dtype=torch.long)

            outputs, logits, features, _ = model(images) # outputs: (1, 16, H, W), logits: (1, 20, H, W), features: (1, 256, H/4, W/4)
            logits = F.interpolate(logits, size=features.shape[-2:], mode='bilinear', align_corners=False) # (1, 20, H/4, W/4)
            features = features[0].detach().cpu().numpy() # (256, H/4, W/4)
            outputs = torch.argmax(outputs, dim=1)[0].detach().cpu().numpy() # (H, W)

            metrics16.update(labels[0].detach().cpu().numpy(), outputs)

            outputs19 = deepcopy(outputs) 
            #outputs19[outputs19 == 13] = 16
            #outputs19[outputs19 == 14] = 17
            #outputs19[outputs19 == 15] = 18
            
            logits = logits[0].detach().cpu().numpy() # (20, H/4, W/4)
            logits = logits[-9:] # (3, H/4, W/4)
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
            metrics19.update(labels_true[0].detach().cpu().numpy(), outputs19)

    score16 = metrics16.get_results()
    score19 = metrics19.get_results()

    print('16 classes')
    print(metrics16.to_str(score16))
    print()
    print('19 classes')
    print(metrics19.to_str(score19))

def select_novel_each_target(novel_loader, unknown_target, device, save_path, shot_num=5):
    print('select novel '+str(unknown_target))
    now_path=os.path.join(save_path,str(unknown_target))
    if (os.path.exists(now_path)==False):
        os.makedirs(now_path)
    file_path=os.path.join(now_path,'novel.txt')
    f = open(file_path,'a',encoding = "utf-8")
    q = queue.PriorityQueue()
    for (images, labels, labels_true, image_name, target_name) in novel_loader:
        
        labels_true=labels_true.to(device, dtype=torch.long)
        now_sum=torch.sum(labels_true==unknown_target).data.cpu()
        q.put([now_sum,(image_name,target_name)])
        if (q.qsize()>shot_num): q.get()
  
    assert q.qsize()==shot_num
    while q.empty()==False:
        now_sum,now_name=q.get()
        image_name="".join(now_name[0])
        target_name="".join(now_name[1])
        f.write(image_name+'\t'+target_name+'\n')
    f.close()

def select_novel(novel_loader, unknown_list, device, save_path='./novel', shot_num=5):
    if (os.path.exists(save_path)==False):
        os.makedirs(save_path)
    for x in unknown_list:
        select_novel_each_target(novel_loader,x,device,save_path, shot_num)
    
def generate_novel(novel_all, novel_path_name, unknown_list, model, device, shot_num=5):
    model.eval()
    novel_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    with torch.no_grad():
        for x in unknown_list:
            print('generate novel: '+str(x))
            log_path=os.path.join(novel_path_name,str(x))
            center=None
            novel_dst = Cityscapes_Novel(novel_path=novel_path_name,novel_no=x, transform=novel_transform)
            novel_loader = data.DataLoader(novel_dst, batch_size=1, shuffle=False, num_workers=4)
            novel_sum=0
            for (image,target) in novel_loader:
                print(image.max(), image.min(), '--------------')
                image=image.to(device)
                target=target.to(device,dtype=torch.long)
                print(image.shape)
                output,feature=model(image)

                if target.shape[-1] != feature.shape[-1]:
                    target = torch.nn.functional.interpolate(target.unsqueeze(dim=1).float(), size=feature.shape[-2:], mode="nearest").squeeze(dim=1)    

                feature=feature.permute(0, 2, 3, 1)
                b,h,w,c=feature.shape
                feature=feature.view(h*w,c)
                target=target.flatten()
                print(target.shape)
                print(feature.shape)

                # for c in range(19):
                #     if c in target:
                #         temp=feature[target==c]
                #     print(c, np.round(np.mean(temp.detach().cpu().numpy(), axis=0), 2))

                feature=feature[target==x]
                feature=torch.sum(feature,dim=0)
                if (center==None): center=torch.zeros(c,).to(device)
                center+=feature
                novel_sum+=torch.sum(target==x)
            center=center/novel_sum
            center_path=os.path.join(log_path,'novel.pth')
            print(center.shape)
            torch.save(center,center_path)
            novel_all[x]=center.clone()
    return novel_all

# def get_novel(center, num_classes, unknown_list):
#     novel = torch.empty((num_classes,center.shape[1]))
#     n=0
#     x=0
#     while (n<num_classes):
#         if n in unknown_list:
#             n+=1
#             continue
#         novel[n]=center[x].clone()
#         x+=1
#         n+=1
#     return novel

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

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
    
    train_dst, val_dst, novel_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=16)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=16)
    novel_loader = data.DataLoader(
        novel_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=16)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3plus_embedding_resnet101': network.deeplabv3plus_embedding_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3plus_metirc_resnet101': network.deeplabv3plus_metirc_resnet101
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = utils.CrossEntropyLoss(ignore_index=255, alpha=0.01, beta=0.01/80, gamma=0)


    # def save_ckpt(path):
    #     """ save current model
    #     """
    #     torch.save({
    #         "cur_itrs": cur_itrs,
    #         "model_state": model.module.state_dict(),
    #         "optimizer_state": optimizer.state_dict(),
    #         "scheduler_state": scheduler.state_dict(),
    #         "best_score": best_score,
    #     }, path)
    #     print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints_131415_embedding')
    # Restore
    # best_score = 0.0
    # cur_itrs = 0
    # cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        res = model.load_state_dict(checkpoint["model_state"])
        print(res)
        #model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        opts.gpu_id = [1]
        # model = nn.DataParallel(model,device_ids=opts.gpu_id)
        #model = nn.DataParallel(model)
        model = model.cuda()

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
    #print(model)
    # if (opts.center):
    #     center=torch.load(opts.center_checkpoint)
    # print(center.shape, opts.num_classes, train_dst.unknown_target, '++++++++++')
    #novel=get_novel(center,opts.num_classes,train_dst.unknown_target)

    novel=np.load(opts.center_checkpoint)
    novel=torch.from_numpy(novel)
    # novel=torch.load('center.pth')
    # novel=torch.cat([novel[:13], torch.zeros((3, novel.shape[1])).float().to(novel.device), novel[13:]], dim=0)
    novel=novel.to(device)
    print(novel.shape)
    #select_novel(novel_loader,train_dst.unknown_target,device)
    novel=generate_novel(novel,'./novel',Cityscapes.unknown_target,model,device,shot_num=5)
    novel=torch.relu(novel)
    for i in range(novel.shape[0]):
        print(i, novel[i].detach().cpu().numpy())
    novel=novel.to(device)
    print(novel.shape)
    # for i in range(novel.shape[0]):
    #     print(i, np.round(novel[i].detach().cpu().numpy(), 2))
    # return
    print('eval mode')
    model.eval()
    val_score, ret_samples = validate(
        opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, novel=novel, ret_samples_ids=vis_sample_id)
    print(metrics.to_str(val_score))
    return

    # if opts.test_only:
    #     model.eval()
    #     val_score, ret_samples = validate(
    #         opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    #     print(metrics.to_str(val_score))
    #     return

    # interval_loss = 0
    # while True: #cur_itrs < opts.total_itrs:
    #     # =====  Train  =====
    #     model.train()
    #     cur_epochs += 1
    #     for (images, labels, labels_true) in train_loader:
    #         cur_itrs += 1

    #         images = images.to(device, dtype=torch.float32)
    #         labels = labels.to(device, dtype=torch.long)

    #         optimizer.zero_grad()
    #         outputs, centers, features = model(images)
    #         loss = criterion(outputs, labels, features)
    #         loss.backward()
    #         optimizer.step()

    #         np_loss = loss.detach().cpu().numpy()
    #         interval_loss += np_loss
    #         if vis is not None:
    #             vis.vis_scalar('Loss', cur_itrs, np_loss)

    #         if (cur_itrs) % 10 == 0:
    #             interval_loss = interval_loss/10
    #             print("Epoch %d, Itrs %d/%d, Loss=%f" %
    #                   (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
    #             interval_loss = 0.0

    #         if (cur_itrs) % opts.val_interval == 0:
    #             save_ckpt('checkpoints_131415_embedding/latest_%s_%s_os%d.pth' %
    #                       (opts.model, opts.dataset, opts.output_stride))
    #             print("validation...")
    #             model.eval()
    #             val_score, ret_samples = validate(
    #                 opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    #             print(metrics.to_str(val_score))
    #             if val_score['Mean IoU'] > best_score:  # save best model
    #                 best_score = val_score['Mean IoU']
    #                 save_ckpt('checkpoints_131415_embedding/best_%s_%s_os%d.pth' %
    #                           (opts.model, opts.dataset,opts.output_stride))

    #             if vis is not None:  # visualize validation score and samples
    #                 vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
    #                 vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
    #                 vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

    #                 for k, (img, target, lbl) in enumerate(ret_samples):
    #                     img = (denorm(img) * 255).astype(np.uint8)
    #                     target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
    #                     lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
    #                     concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
    #                     vis.vis_image('Sample %d' % k, concat_img)
    #             model.train()
    #         scheduler.step()  

    #         if cur_itrs >=  opts.total_itrs:
    #             return

        
if __name__ == '__main__':
    main()
