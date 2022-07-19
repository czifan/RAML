import json
import os
from collections import namedtuple

from matplotlib import set_loglevel

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
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
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    unknown_target = None
    # unknown_target = [1, 3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 18]
    # 12+7
    unknown_target = [10,13,14,15,16,17,18]
    # 14+5
    # unknown_target = [10,13,14,15,16]
    # 18+1
    #unknown_target = [13]
    # 16+3 / 16+1
    #unknown_target = [13,14,15]
    # unknown_target = [i for i in range(19)]
    # unknown_target.pop(13)
    print('unknown_target is : ', unknown_target)
    # unknown_target = [18]
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        # self.targets_dir = self.images_dir

        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []


        if split not in ['train', 'test_car', 'val','test_truck', 'test_bus', 'test_car_1_shot',
                         'test_truck_1_shot', 'test_bus_1_shot', 'car_vis', 'bus_vis','demo_video',
                         'car_100','car_1000']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            files_name = os.listdir(img_dir)
            files_name = sorted(files_name)
            for file_name in files_name:
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):

        target = cls.id_to_train_id[np.array(target)]
        target_true = target.copy()
        # instance, counts = np.unique(target, False, False, True)
        # print('target', instance, counts)
        if cls.unknown_target != None:
            cont = 0
            for h_c in cls.unknown_target:

                target[target == h_c - cont] = 100
                for c in range(h_c - cont + 1, 19):
                    target[target == c] = c - 1
                    # target_true[target_true == c] = c - 1
                cont = cont + 1
            # target_true[target == 100] = 19 - len(cls.unknown_target)
            target[target == 100] = 255

        return target, target_true

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        # image = Image.open(self.images[index])
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target, target_true = self.encode_target(target)
        target_lst, class_lst = self.encode_target_czifan(target)
        
        return image, target, target_true, target_lst, class_lst

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)

    def encode_target_czifan(self, target, output_size=16):
        known_class = 19 - len(Cityscapes.unknown_target)
        target_lst = np.zeros((known_class + 1, *target.shape))
        class_lst = np.ones(known_class + 1) * 255
        for c in range(known_class):
            target_lst[c] = (target == c)
            class_lst[c] = c 
        return target_lst.astype(np.uint8), class_lst.astype(np.uint8)

        # target_lst = np.zeros((output_size**2, *target.shape))
        # class_lst = np.ones(output_size**2) * 255
        # for t in np.unique(target):
        #     tmp = np.where(target == t)
        #     gy, gx = int(np.mean(tmp[0])/32), int(np.mean(tmp[1])/32)
        #     target_lst[gy*output_size+gx,...] = (target == t)
        #     class_lst[gy*output_size+gx] = t 
        # return target_lst.astype(np.uint8), class_lst.astype(np.uint8)

        # temp = cv2.resize(target.astype(np.uint8), (output_size, output_size), interpolation=cv2.INTER_LINEAR).reshape(-1)
        # #temp = torch.nn.functional.interpolate(target.clone().unsqueeze(dim=1).float(), size=[output_size, output_size], mode="nearest").view(-1)
        # target_lst, class_lst = [], []
        # for t in temp:
        #     if t == 255:
        #         target_lst.append(np.zeros_like(target))
        #     else:
        #         target_lst.append(target == t)
        #     class_lst.append(t.item())
        # target_lst = np.stack(target_lst, axis=0).astype(np.uint8) # (256, 512, 512)
        # class_lst = np.asarray(class_lst).astype(np.uint8) # (256,)
        # return target_lst, class_lst
